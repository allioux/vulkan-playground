use std::sync::Arc;
use std::{error::Error, ffi};

use ash::khr::get_physical_device_properties2;
use ash::prelude::VkResult;
use ash::{ext::debug_utils, vk, Entry, Instance};
use simple_error::bail;
use winit::dpi::PhysicalSize;
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

pub use device_data::DeviceData;
pub use surface_data::SurfaceData;
pub use swapchain_data::SwapchainData;

mod device_data;
mod surface_data;
mod swapchain_data;

pub struct PipelineDescriptor<'a> {
    pub pipeline_layout: vk::PipelineLayout,
    pub vertex_input_info: vk::PipelineVertexInputStateCreateInfo<'a>,
    pub shader_stages_info: &'a [vk::PipelineShaderStageCreateInfo<'a>],
}

pub struct ImageDescriptor {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

#[derive(Default)]
pub struct TransitionImageLayoutDesc {
    pub format: vk::Format,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_access_mask: vk::AccessFlags,
    pub dst_access_mask: vk::AccessFlags,
    pub src_stage_mask: vk::PipelineStageFlags,
    pub dst_stage_mask: vk::PipelineStageFlags,
}

impl TransitionImageLayoutDesc {
    pub fn from_layouts(
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> Result<TransitionImageLayoutDesc, Box<dyn Error>> {
        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            Ok(TransitionImageLayoutDesc {
                format,
                old_layout,
                new_layout,
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
                dst_stage_mask: vk::PipelineStageFlags::TRANSFER,
                ..Default::default()
            })
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::READ_ONLY_OPTIMAL
        {
            Ok(TransitionImageLayoutDesc {
                format,
                old_layout,
                new_layout,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                src_stage_mask: vk::PipelineStageFlags::TRANSFER,
                dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            })
        } else if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        {
            Ok(TransitionImageLayoutDesc {
                format,
                old_layout,
                new_layout,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                src_stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ..Default::default()
            })
        } else if old_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            && new_layout == vk::ImageLayout::PRESENT_SRC_KHR
        {
            Ok(TransitionImageLayoutDesc {
                format,
                old_layout,
                new_layout,
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                ..Default::default()
            })
        } else {
            Err("Image layout transition not supported.".into())
        }
    }
}

pub struct Base {
    pub instance: Instance,
    pub surface_data: SurfaceData,
    pub device_data: DeviceData,
    pub present_queue: vk::Queue,
    pub swapchain_data: SwapchainData,
}

impl Base {
    pub fn new(window: Arc<Window>) -> Result<Base, Box<dyn Error>> {
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
                ref surface_format_khr,
                surface_extent,
                ..
            } = device_data;

            let present_queue = device.get_device_queue(*queue_family_index as u32, 0);

            let swapchain_data = SwapchainData::new(
                &instance,
                &device_data,
                physical_device,
                &surface_extension,
                *surface,
                surface_extent,
                surface_format_khr,
            )?;

            Ok(Base {
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
                None => self.device_data.surface_extent,
            };

            let surface_capabilities = self
                .surface_data
                .surface_extension
                .get_physical_device_surface_capabilities(
                    self.device_data.physical_device,
                    self.surface_data.surface,
                )?;

            let surface_extent = match surface_capabilities.current_extent.width {
                u32::MAX => extent,
                _ => surface_capabilities.current_extent,
            };

            self.device_data.surface_extent = surface_extent;

            self.swapchain_data = SwapchainData::new(
                &self.instance,
                &self.device_data,
                &self.device_data.physical_device,
                &self.surface_data.surface_extension,
                self.surface_data.surface,
                surface_extent,
                &self.device_data.surface_format_khr,
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
        depth_image: vk::Image,
        depth_image_view: vk::ImageView,
        graphics_pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        vertex_buffers: &[vk::Buffer],
        index_buffer: vk::Buffer,
        index_type: vk::IndexType,
        index_count: u32,
        descriptor_sets: &[vk::DescriptorSet],
        frame_index: u32,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            let DeviceData {
                ref device,
                surface_extent,
                ref dynamic_rendering_khr,
                ..
            } = self.device_data;

            let command_buffer_begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::empty());

            device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
            device.cmd_bind_vertex_buffers(command_buffer, 0, vertex_buffers, &[0]);
            device.cmd_bind_index_buffer(command_buffer, index_buffer, 0, index_type);

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(surface_extent.width as f32)
                .height(surface_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            device.cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

            let scissor: vk::Rect2D = surface_extent.into();
            device.cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

            let color_attachments = [vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .resolve_mode(vk::ResolveModeFlags::NONE)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue::default())];

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_image_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue::default().depth(1.0),
                });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D::default())
                        .extent(
                            vk::Extent2D::default()
                                .height(surface_extent.height)
                                .width(surface_extent.width),
                        ),
                )
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment)
                .layer_count(1);

            self.transition_image_layout(
                command_buffer,
                image,
                TransitionImageLayoutDesc::from_layouts(
                    vk::Format::R8G8B8A8_SRGB,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )?,
            );

            dynamic_rendering_khr.cmd_begin_rendering(command_buffer, &rendering_info);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[descriptor_sets[frame_index as usize]],
                &[],
            );

            device.cmd_draw_indexed(command_buffer, index_count, 1, 0, 0, 0);
            dynamic_rendering_khr.cmd_end_rendering(command_buffer);

            self.transition_image_layout(
                command_buffer,
                image,
                TransitionImageLayoutDesc::from_layouts(
                    vk::Format::R8G8B8A8_SRGB,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                )?,
            );

            device.end_command_buffer(command_buffer)?;

            Ok(())
        }
    }

    pub fn create_pipeline(
        &self,
        pipeline_desc: &PipelineDescriptor,
    ) -> Result<vk::Pipeline, Box<dyn Error>> {
        unsafe {
            let device = &self.device_data.device;

            let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .depth_bias_constant_factor(0.0)
                .depth_bias_clamp(0.0)
                .depth_bias_slope_factor(0.0);

            let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .min_sample_shading(1.0)
                .alpha_to_coverage_enable(false)
                .alpha_to_one_enable(false);

            let color_blend_attachment_info = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD);

            let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(std::slice::from_ref(&color_blend_attachment_info))
                .blend_constants([0.0, 0.0, 0.0, 0.0]);

            let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(std::slice::from_ref(
                    &self.device_data.surface_format_khr.format,
                ))
                .depth_attachment_format(self.find_depth_format()?);

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false);

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&pipeline_desc.shader_stages_info)
                .vertex_input_state(&pipeline_desc.vertex_input_info)
                .input_assembly_state(&input_assembly_info)
                .viewport_state(&viewport_state_info)
                .rasterization_state(&rasterizer_info)
                .multisample_state(&multisampling_info)
                .color_blend_state(&color_blending_info)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_desc.pipeline_layout)
                .subpass(0)
                .depth_stencil_state(&depth_stencil_info)
                .push_next(&mut pipeline_rendering_info);

            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .unwrap()[0];

            Ok(pipeline)
        }
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, vk::MemoryRequirements), Box<dyn Error>> {
        unsafe {
            let device = &self.device_data.device;

            let buffer_info = vk::BufferCreateInfo::default()
                .size(size as u64)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = device.create_buffer(&buffer_info, None)?;

            let memory_requirements = device.get_buffer_memory_requirements(buffer);

            let memory_type_index =
                self.find_memory_type(memory_requirements.memory_type_bits, properties)?;

            let memory_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(memory_requirements.size)
                .memory_type_index(memory_type_index);

            let buffer_memory = device.allocate_memory(&memory_alloc_info, None)?;
            device.bind_buffer_memory(buffer, buffer_memory, 0)?;

            Ok((buffer, buffer_memory, memory_requirements))
        }
    }

    pub fn create_image(
        &self,
        image_descriptor: ImageDescriptor,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::MemoryRequirements), Box<dyn Error>> {
        unsafe {
            let device = &self.device_data.device;

            let ImageDescriptor {
                extent,
                tiling,
                format,
                usage,
                properties,
            } = image_descriptor;

            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .format(format)
                .tiling(tiling)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(vk::SampleCountFlags::TYPE_1);

            let image = device.create_image(&image_info, None)?;

            let memory_requirements = device.get_image_memory_requirements(image);

            let allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(memory_requirements.size)
                .memory_type_index(
                    self.find_memory_type(memory_requirements.memory_type_bits, properties)?,
                );

            let image_memory = device.allocate_memory(&allocate_info, None)?;
            device.bind_image_memory(image, image_memory, 0)?;

            Ok((image, image_memory, memory_requirements))
        }
    }

    pub fn create_and_populate_buffer<T>(
        &self,
        command_pool: vk::CommandPool,
        usage: vk::BufferUsageFlags,
        src: Vec<T>,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
        unsafe {
            let device = &self.device_data.device;

            let buffer_size = (src.len() * size_of::<T>()) as u64;

            let (staging_buffer, staging_buffer_memory, _) = self.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let dst = device.map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;

            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.cast(), src.len());

            device.unmap_memory(staging_buffer_memory);

            let (buffer, buffer_memory, _) = self.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | usage,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            self.copy_buffer(command_pool, staging_buffer, buffer, buffer_size)?;
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);

            Ok((buffer, buffer_memory))
        }
    }

    pub fn begin_single_time_commands(
        &self,
        command_pool: vk::CommandPool,
    ) -> VkResult<vk::CommandBuffer> {
        unsafe {
            let device = &self.device_data.device;

            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1);

            let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(command_buffer, &begin_info)?;

            Ok(command_buffer)
        }
    }

    pub fn end_single_time_commands(
        &self,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
    ) -> VkResult<()> {
        unsafe {
            let device = &self.device_data.device;
            device.end_command_buffer(command_buffer)?;

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

            device.queue_submit(self.present_queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(self.present_queue)?;
            device.free_command_buffers(command_pool, std::slice::from_ref(&command_buffer));

            Ok(())
        }
    }

    pub fn copy_buffer(
        &self,
        command_pool: vk::CommandPool,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) -> VkResult<()> {
        unsafe {
            let command_buffer = self.begin_single_time_commands(command_pool)?;

            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(size);

            self.device_data.device.cmd_copy_buffer(
                command_buffer,
                src_buffer,
                dst_buffer,
                &[region],
            );

            self.end_single_time_commands(command_pool, command_buffer)
        }
    }

    pub fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        transition_image_layout_desc: TransitionImageLayoutDesc,
    ) -> () {
        let TransitionImageLayoutDesc {
            old_layout,
            new_layout,
            src_access_mask,
            dst_access_mask,
            src_stage_mask,
            dst_stage_mask,
            ..
        } = transition_image_layout_desc;

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let image_memory_barrier = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .image(image)
            .subresource_range(subresource_range);

        unsafe {
            self.device_data.device.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&image_memory_barrier),
            )
        }
    }

    pub fn copy_buffer_to_image(
        &self,
        command_pool: vk::CommandPool,
        buffer: vk::Buffer,
        image: vk::Image,
        image_extent: vk::Extent3D,
    ) -> VkResult<()> {
        let command_buffer = self.begin_single_time_commands(command_pool)?;

        let image_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(image_subresource)
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(image_extent);

        unsafe {
            self.device_data.device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        self.end_single_time_commands(command_pool, command_buffer)
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

    pub fn find_supported_format(
        &self,
        candidates: Vec<vk::Format>,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Result<vk::Format, Box<dyn Error>> {
        unsafe {
            for candidate in candidates {
                let props = self.instance.get_physical_device_format_properties(
                    self.device_data.physical_device,
                    candidate,
                );

                if tiling == vk::ImageTiling::LINEAR
                    && (props.linear_tiling_features & features) == features
                {
                    return Ok(candidate);
                } else if tiling == vk::ImageTiling::OPTIMAL
                    && (props.optimal_tiling_features & features) == features
                {
                    return Ok(candidate);
                }
            }
            bail!("Failed to find a supported format.")
        }
    }

    pub fn find_depth_format(&self) -> Result<vk::Format, Box<dyn Error>> {
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        self.find_supported_format(
            candidates,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    pub fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    pub fn create_depth_image(
        &self,
    ) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory), Box<dyn Error>> {
        let depth_format = self.find_depth_format()?;
        let (depth_image, depth_image_memory, _) = self.create_image(ImageDescriptor {
            extent: self.device_data.surface_extent.into(),
            format: depth_format,
            tiling: vk::ImageTiling::OPTIMAL,
            properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        })?;

        let depth_image_view = self.device_data.create_image_view(
            depth_image,
            depth_format,
            vk::ImageAspectFlags::DEPTH,
        )?;

        Ok((depth_image, depth_image_view, depth_image_memory))
    }
}

impl Drop for Base {
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
