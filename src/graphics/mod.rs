use std::sync::Arc;
use std::{error::Error, ffi};

use ash::khr::get_physical_device_properties2;
use ash::prelude::VkResult;
use ash::{ext::debug_utils, vk, Entry, Instance};
use winit::dpi::PhysicalSize;
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

pub use device_data::DeviceData;
pub use surface_data::SurfaceData;
pub use swapchain_data::SwapchainData;

mod device_data;
mod surface_data;
mod swapchain_data;

pub struct Graphics {
    pub instance: Instance,
    pub surface_data: SurfaceData,
    pub device_data: DeviceData,
    pub present_queue: vk::Queue,
    pub swapchain_data: SwapchainData,
}

impl Graphics {
    pub fn new(window: Arc<Window>) -> Result<Graphics, Box<dyn Error>> {
        unsafe {
            let app_info =
                vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 283));

            let layers_names_raw = if cfg!(debug_assertions) {
                let layer_names = [ffi::CStr::from_bytes_with_nul_unchecked(
                    b"VK_LAYER_KHRONOS_validation\0",
                )];

                layer_names
                    .iter()
                    .map(|raw_name| raw_name.as_ptr())
                    .collect()
            } else {
                vec![]
            };

            let mut extension_names =
                ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())
                    .unwrap()
                    .to_vec();
            extension_names.push(debug_utils::NAME.as_ptr());
            extension_names.push(get_physical_device_properties2::NAME.as_ptr());
            #[cfg(target_os = "macos")]
            extension_names.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());

            let flags = if cfg!(target_os = "macos") {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            };

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names)
                .flags(flags);

            let entry = Entry::linked();

            let instance = entry.create_instance(&create_info, None)?;

            let surface_data = SurfaceData::new(&entry, &instance, &window)?;

            let SurfaceData {
                ref surface,
                ref surface_extension,
                ..
            } = surface_data;

            let window_size = window.inner_size();

            let extent = vk::Extent2D {
                height: window_size.height,
                width: window_size.width,
            };

            let device_data = DeviceData::new(&instance, *surface, &surface_extension, extent)?;

            let DeviceData {
                ref device,
                ref physical_device,
                ref queue_family_index,
                ref surface_format,
                surface_resolution,
                ..
            } = device_data;

            let present_queue = device.get_device_queue(*queue_family_index as u32, 0);

            let swapchain_data = SwapchainData::new(
                &instance,
                &device,
                physical_device,
                &surface_extension,
                *surface,
                surface_resolution,
                surface_format,
            )?;

            Ok(Graphics {
                instance,
                surface_data,
                device_data,
                present_queue,
                swapchain_data,
            })
        }
    }

    pub fn recreate_swapchain(
        &mut self,
        window_size: Option<PhysicalSize<u32>>,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            let _ = self.device_data.device.device_wait_idle();
            self.cleanup_swapchain();

            let extent = match window_size {
                Some(size) => vk::Extent2D {
                    height: size.height,
                    width: size.width,
                },
                None => self.device_data.surface_resolution,
            };

            let surface_capabilities = self
                .surface_data
                .surface_extension
                .get_physical_device_surface_capabilities(
                    self.device_data.physical_device,
                    self.surface_data.surface,
                )?;

            let surface_resolution = match surface_capabilities.current_extent.width {
                u32::MAX => extent,
                _ => surface_capabilities.current_extent,
            };

            self.device_data.surface_resolution = surface_resolution;

            self.swapchain_data = SwapchainData::new(
                &self.instance,
                &self.device_data.device,
                &self.device_data.physical_device,
                &self.surface_data.surface_extension,
                self.surface_data.surface,
                surface_resolution,
                &self.device_data.surface_format,
            )?;

            Ok(())
        }
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            for &view in &self.swapchain_data.image_views {
                self.device_data.device.destroy_image_view(view, None)
            }
            self.swapchain_data
                .swapchain_extension
                .destroy_swapchain(self.swapchain_data.swapchain, None);
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        image_view: vk::ImageView,
        graphics_pipeline: vk::Pipeline,
        buffers: &[vk::Buffer],
    ) -> VkResult<()> {
        unsafe {
            let DeviceData {
                ref device,
                surface_resolution,
                ref dynamic_rendering_extension,
                ..
            } = self.device_data;

            let command_buffer_begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::empty());

            device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

            device.cmd_bind_vertex_buffers(command_buffer, 0, buffers, &[0]);

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(surface_resolution.width as f32)
                .height(surface_resolution.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

            let scissor: vk::Rect2D = surface_resolution.into();
            device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

            let color_attachments = [vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .resolve_mode(vk::ResolveModeFlags::NONE)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue::default())];

            let rendering_info = vk::RenderingInfo::default()
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D::default())
                        .extent(
                            vk::Extent2D::default()
                                .height(surface_resolution.height)
                                .width(surface_resolution.width),
                        ),
                )
                .color_attachments(&color_attachments)
                .layer_count(1);

            let image_memory_barrier = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&image_memory_barrier),
            );

            dynamic_rendering_extension.cmd_begin_rendering(command_buffer, &rendering_info);
            device.cmd_draw(command_buffer, 3 as u32, 1, 0, 0);
            dynamic_rendering_extension.cmd_end_rendering(command_buffer);

            let image_memory_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&image_memory_barrier),
            );

            device.end_command_buffer(command_buffer)
        }
    }

    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        unsafe {
            let memory_properties = self
                .instance
                .get_physical_device_memory_properties(self.device_data.physical_device);

            for i in 0..memory_properties.memory_type_count {
                if ((type_filter & (1 << i)) != 0)
                    && (memory_properties.memory_types[i as usize].property_flags & properties)
                        == properties
                {
                    return Ok(i);
                }
            }
            Err("Failed to find an appropriate memory type.".to_string())
        }
    }
}

impl Drop for Graphics {
    fn drop(&mut self) {
        unsafe {
            self.cleanup_swapchain();
            self.surface_data
                .surface_extension
                .destroy_surface(self.surface_data.surface, None);
            self.device_data.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
