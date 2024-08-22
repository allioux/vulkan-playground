use std::error::Error;
use std::sync::Arc;

use ash::prelude::VkResult;
use ash::util::read_spv;
use ash::vk;
use winit::window::Window;

use crate::graphical_system::GraphicalSystem;

pub struct Triangle {
    graphical_system: GraphicalSystem,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

impl Triangle {
    fn create_shader_module(
        graphical_system: &GraphicalSystem,
        code: Vec<u32>,
    ) -> VkResult<vk::ShaderModule> {
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);

        unsafe {
            graphical_system
                .device
                .create_shader_module(&shader_info, None)
        }
    }

    pub fn new(window: Arc<Window>) -> Result<Triangle, Box<dyn Error>> {
        unsafe {
            let graphical_system = GraphicalSystem::new(window)?;

            // PIPELINE

            let mut vert_shader = std::fs::File::open("shaders/shader_vert.spv").unwrap();
            let mut frag_shader = std::fs::File::open("shaders/shader_frag.spv").unwrap();

            let vert_code = read_spv(&mut vert_shader).unwrap();
            let vert_reflect = spirv_reflect::ShaderModule::load_u32_data(&vert_code).unwrap();

            let frag_code = read_spv(&mut frag_shader).unwrap();
            let frag_reflect = spirv_reflect::ShaderModule::load_u32_data(&frag_code).unwrap();

            let vert_module = Self::create_shader_module(&graphical_system, vert_code).unwrap();
            let frag_module = Self::create_shader_module(&graphical_system, frag_code).unwrap();

            let vert_entry_name =
                std::ffi::CString::new(vert_reflect.get_entry_point_name()).unwrap();
            let frag_entry_name =
                std::ffi::CString::new(frag_reflect.get_entry_point_name()).unwrap();

            let shader_stages_info = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_module)
                    .name(vert_entry_name.as_c_str()),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_module)
                    .name(frag_entry_name.as_c_str()),
            ];

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

            let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(graphical_system.surface_resolution.width as f32)
                .height(graphical_system.surface_resolution.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);

            let scissors = [graphical_system.surface_resolution.into()];

            let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
                .viewports(std::slice::from_ref(&viewport))
                .scissors(&scissors);

            let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
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

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[])
                .push_constant_ranges(&[]);

            let pipeline_layout = graphical_system
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)?;

            let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(std::slice::from_ref(
                    &graphical_system.surface_format.format,
                ));

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages_info)
                .vertex_input_state(&vertex_input_info)
                .input_assembly_state(&input_assembly_info)
                .viewport_state(&viewport_state_info)
                .rasterization_state(&rasterizer_info)
                .multisample_state(&multisampling_info)
                .color_blend_state(&color_blending_info)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .subpass(0)
                .push_next(&mut pipeline_rendering_info);

            let pipelines = graphical_system
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .unwrap();
         
            // COMMAND POOL

            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(graphical_system.queue_family_index);

            let command_pool = graphical_system
                .device
                .create_command_pool(&command_pool_info, None)?;

            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = graphical_system
                .device
                .allocate_command_buffers(&allocate_info)?[0];

            // SYNC OBJECTS

            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let image_available_semaphore = graphical_system
                .device
                .create_semaphore(&semaphore_info, None)?;
            let render_finished_semaphore = graphical_system
                .device
                .create_semaphore(&semaphore_info, None)?;
            let in_flight_fence = graphical_system.device.create_fence(&fence_info, None)?;

            Ok(Triangle {
                graphical_system,
                vert_module,
                frag_module,
                pipeline_layout,
                pipelines,
                command_pool,
                command_buffer,
                image_available_semaphore,
                render_finished_semaphore,
                in_flight_fence,
            })
        }
    }

    pub fn draw_frame(&self) {
        unsafe {
            let _ = self.graphical_system.device.wait_for_fences(
                std::slice::from_ref(&self.in_flight_fence),
                true,
                std::u64::MAX,
            );
            let _ = self
                .graphical_system
                .device
                .reset_fences(std::slice::from_ref(&self.in_flight_fence));

            let (image_index, _) = self
                .graphical_system
                .swapchain_loader
                .acquire_next_image(
                    self.graphical_system.swapchain,
                    std::u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire the next image.");

            self.graphical_system
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset the command buffer.");

            self.graphical_system
                .record_command_buffer(
                    self.command_buffer,
                    self.graphical_system.swapchain_images[image_index as usize],
                    self.graphical_system.swapchain_image_views[image_index as usize],
                    self.pipelines[0],
                )
                .expect("Failed to record the command buffer.");

            let signal_semaphores = [self.render_finished_semaphore];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(std::slice::from_ref(&self.image_available_semaphore))
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(std::slice::from_ref(&self.command_buffer))
                .signal_semaphores(&signal_semaphores);

            self.graphical_system
                .device
                .queue_submit(
                    self.graphical_system.present_queue,
                    &[submit_info],
                    self.in_flight_fence,
                )
                .expect("Failed to submit draw command buffer.");

            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(std::slice::from_ref(&self.graphical_system.swapchain))
                .image_indices(std::slice::from_ref(&image_index));

            self.graphical_system
                .swapchain_loader
                .queue_present(self.graphical_system.present_queue, &present_info)
                .expect("");
        }
    }
}

impl Drop for Triangle {
    fn drop(&mut self) {
        unsafe {
            let _ = self.graphical_system.device.device_wait_idle();
            self.graphical_system
                .device
                .destroy_fence(self.in_flight_fence, None);
            self.graphical_system
                .device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.graphical_system
                .device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.graphical_system
                .device
                .destroy_command_pool(self.command_pool, None);
            self.pipelines.iter().for_each(|&pipeline| {
                self.graphical_system
                    .device
                    .destroy_pipeline(pipeline, None)
            });
            self.graphical_system
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.graphical_system
                .device
                .destroy_shader_module(self.vert_module, None);
            self.graphical_system
                .device
                .destroy_shader_module(self.frag_module, None);
        }
    }
}
