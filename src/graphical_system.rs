use std::sync::Arc;
use std::{error::Error, ffi};

use ash::khr::{
    create_renderpass2, depth_stencil_resolve, dynamic_rendering, get_physical_device_properties2,
    maintenance2, multiview, surface, swapchain,
};
use ash::prelude::VkResult;
use ash::{ext::debug_utils, vk, Device, Entry, Instance};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

pub struct GraphicalSystem {
    pub instance: Instance,
    pub surface_loader: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
    pub device: Device,
    pub dynamic_device: dynamic_rendering::Device,
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,
    pub swapchain_loader: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
}

impl GraphicalSystem {
    pub fn new(window: Arc<Window>) -> Result<GraphicalSystem, Box<dyn Error>> {
        unsafe {
            // INSTANCE

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

            // PHYSICAL DEVICE && SURFACE

            let physical_devices = instance.enumerate_physical_devices()?;

            let surface_loader = surface::Instance::new(&entry, &instance);

            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .unwrap();

            let (physical_device, queue_family_index) = physical_devices
                .iter()
                .find_map(|device| {
                    instance
                        .get_physical_device_queue_family_properties(*device)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *device,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((*device, index))
                            } else {
                                None
                            }
                        })
                })
                .ok_or("Could not find a suitable device.")?;

            // DEVICE

            let device_extension_names_raw = [
                swapchain::NAME.as_ptr(),
                multiview::NAME.as_ptr(),
                maintenance2::NAME.as_ptr(),
                create_renderpass2::NAME.as_ptr(),
                depth_stencil_resolve::NAME.as_ptr(),
                dynamic_rendering::NAME.as_ptr(),
                #[cfg(any(target_os = "macos"))]
                ash::khr::portability_subset::NAME.as_ptr(),
            ];

            let device_features = vk::PhysicalDeviceFeatures::default();

            let mut dynamic_rendering_feature =
                vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index as u32)
                .queue_priorities(&[1.0]);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(&mut dynamic_rendering_feature);

            let device = instance.create_device(physical_device, &device_create_info, None)?;

            let dynamic_device = dynamic_rendering::Device::new(&instance, &device);

            // QUEUE

            let present_queue = device.get_device_queue(queue_family_index as u32, 0);

            // SWAPCHAIN

            let surface_formats =
                surface_loader.get_physical_device_surface_formats(physical_device, surface)?;

            let default_format = surface_formats[0].clone();

            let surface_format: vk::SurfaceFormatKHR = *surface_formats
                .into_iter()
                .filter(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_SRGB
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .collect::<Vec<_>>()
                .first()
                .unwrap_or(&default_format);

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)?;

            let mut desired_image_count = surface_capabilities.min_image_count + 1;

            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }

            let window_size = window.inner_size();

            let surface_resolution = match surface_capabilities.current_extent.width {
                u32::MAX => vk::Extent2D {
                    width: window_size.width,
                    height: window_size.height,
                },
                _ => surface_capabilities.current_extent,
            };

            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };

            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)?;

            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_loader = swapchain::Device::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None)?;

            let swapchain_images = swapchain_loader.get_swapchain_images(swapchain)?;

            let swapchain_image_views = swapchain_images
                .iter()
                .map(|&image| {
                    let info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(
                            vk::ComponentMapping::default()
                                .r(vk::ComponentSwizzle::IDENTITY)
                                .g(vk::ComponentSwizzle::IDENTITY)
                                .b(vk::ComponentSwizzle::IDENTITY)
                                .a(vk::ComponentSwizzle::IDENTITY),
                        )
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        );

                    device.create_image_view(&info, None)
                })
                .collect::<Result<_, _>>()?;

            Ok(GraphicalSystem {
                instance,
                surface_loader,
                surface,
                surface_format,
                surface_resolution,
                device,
                dynamic_device,
                queue_family_index: queue_family_index as u32,
                present_queue,
                swapchain_loader,
                swapchain,
                swapchain_images,
                swapchain_image_views,
            })
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        image_view: vk::ImageView,
        graphics_pipeline: vk::Pipeline,
    ) -> VkResult<()> {
        unsafe {
            let command_buffer_begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::empty());

            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(self.surface_resolution.width as f32)
                .height(self.surface_resolution.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            self.device
                .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

            let scissor = self.surface_resolution.into();
            self.device
                .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

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
                                .height(self.surface_resolution.height)
                                .width(self.surface_resolution.width),
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

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&image_memory_barrier),
            );

            self.dynamic_device
                .cmd_begin_rendering(command_buffer, &rendering_info);
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            self.dynamic_device.cmd_end_rendering(command_buffer);

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

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&image_memory_barrier),
            );

            self.device.end_command_buffer(command_buffer)
        }
    }
}

impl Drop for GraphicalSystem {
    fn drop(&mut self) {
        unsafe {
            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None)
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
