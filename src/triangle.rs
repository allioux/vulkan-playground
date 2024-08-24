use std::error::Error;
use std::mem::offset_of;
use std::sync::Arc;

use ash::prelude::VkResult;
use ash::util::read_spv;
use ash::vk;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::base::{Base, DeviceData, PipelineDescriptor};

#[repr(C)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32),
        ]
    }
}

pub struct Triangle {
    base: Base,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    in_flight_frames: u32,
    current_frame: u32,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
}

impl Triangle {
    fn create_shader_module(base: &Base, code: Vec<u32>) -> VkResult<vk::ShaderModule> {
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);

        unsafe {
            base.device_data
                .device
                .create_shader_module(&shader_info, None)
        }
    }

    pub fn new(window: Arc<Window>) -> Result<Triangle, Box<dyn Error>> {
        let in_flight_frames = 2;

        let vertices = vec![
            Vertex {
                pos: [0.0, -0.5],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                pos: [0.5, 0.5],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                pos: [-0.5, 0.5],
                color: [0.0, 0.0, 1.0],
            },
        ];

        unsafe {
            let base = Base::new(window)?;

            let DeviceData {
                ref device,
                queue_family_index,
                ..
            } = base.device_data;

            let mut vert_shader = std::fs::File::open("shaders/shader_vert.spv").unwrap();
            let mut frag_shader = std::fs::File::open("shaders/shader_frag.spv").unwrap();

            let vert_code = read_spv(&mut vert_shader).unwrap();
            let vert_reflect = spirv_reflect::ShaderModule::load_u32_data(&vert_code).unwrap();

            let frag_code = read_spv(&mut frag_shader).unwrap();
            let frag_reflect = spirv_reflect::ShaderModule::load_u32_data(&frag_code).unwrap();

            let vert_module = Self::create_shader_module(&base, vert_code).unwrap();
            let frag_module = Self::create_shader_module(&base, frag_code).unwrap();

            let vert_entry_name =
                std::ffi::CString::new(vert_reflect.get_entry_point_name()).unwrap();
            let frag_entry_name =
                std::ffi::CString::new(frag_reflect.get_entry_point_name()).unwrap();

            let vertex_binding_description = Vertex::get_binding_description();
            let vertex_attribute_descriptions = Vertex::get_attribute_descriptions();
            let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(std::slice::from_ref(&vertex_binding_description))
                .vertex_attribute_descriptions(&vertex_attribute_descriptions);

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

            // PIPELINE

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[])
                .push_constant_ranges(&[]);

            let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

            let pipeline = base.create_pipeline(&PipelineDescriptor {
                pipeline_layout,
                vertex_input_info,
                shader_stages_info: &shader_stages_info,
            })?;

            // COMMAND POOL

            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let command_pool = device.create_command_pool(&command_pool_info, None)?;

            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(in_flight_frames);

            let command_buffers = device.allocate_command_buffers(&allocate_info)?;

            // VERTEX BUFFERS

            let buffer_size = (size_of::<Vertex>() * vertices.len()) as u64;

            let (staging_buffer, staging_buffer_memory) = base.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let data = device.map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;

            std::ptr::copy_nonoverlapping(vertices.as_ptr(), data.cast(), vertices.len());

            let _ = device.unmap_memory(staging_buffer_memory);

            let (vertex_buffer, vertex_buffer_memory) = base.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            base.copy_buffer(command_pool, staging_buffer, vertex_buffer, buffer_size)?;
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);

            // SYNC OBJECTS

            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

            let mut image_available_semaphores = vec![];
            let mut render_finished_semaphores = vec![];
            let mut in_flight_fences = vec![];

            for _ in 0..in_flight_frames {
                image_available_semaphores.push(device.create_semaphore(&semaphore_info, None)?);

                render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);

                in_flight_fences.push(device.create_fence(&fence_info, None)?);
            }

            Ok(Triangle {
                base,
                vert_module,
                frag_module,
                pipeline_layout,
                pipelines: vec![pipeline],
                command_pool,
                command_buffers,
                image_available_semaphores,
                render_finished_semaphores,
                in_flight_fences,
                in_flight_frames,
                current_frame: 0,
                vertex_buffer,
                vertex_buffer_memory,
            })
        }
    }

    pub fn draw_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let frame_index = self.current_frame as usize;
        let DeviceData { device, .. } = &self.base.device_data;
        unsafe {
            let _ = device.wait_for_fences(
                std::slice::from_ref(&self.in_flight_fences[frame_index]),
                true,
                std::u64::MAX,
            );

            device.reset_command_buffer(
                self.command_buffers[frame_index],
                vk::CommandBufferResetFlags::empty(),
            )?;

            let acquire_image_result = self
                .base
                .swapchain_data
                .swapchain_extension
                .acquire_next_image(
                    self.base.swapchain_data.swapchain,
                    std::u64::MAX,
                    self.image_available_semaphores[frame_index],
                    vk::Fence::null(),
                );

            match acquire_image_result {
                Ok((image_index, _)) => {
                    let _ = device
                        .reset_fences(std::slice::from_ref(&self.in_flight_fences[frame_index]))?;

                    self.base
                        .record_command_buffer(
                            self.command_buffers[frame_index],
                            self.base.swapchain_data.images[image_index as usize],
                            self.base.swapchain_data.image_views[image_index as usize],
                            self.pipelines[0],
                            &[self.vertex_buffer],
                        )
                        .expect("Failed to record the command buffer.");

                    let signal_semaphores = [self.render_finished_semaphores[frame_index]];

                    let submit_info = vk::SubmitInfo::default()
                        .wait_semaphores(std::slice::from_ref(
                            &self.image_available_semaphores[frame_index],
                        ))
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(std::slice::from_ref(&self.command_buffers[frame_index]))
                        .signal_semaphores(&signal_semaphores);

                    device
                        .queue_submit(
                            self.base.present_queue,
                            &[submit_info],
                            self.in_flight_fences[frame_index],
                        )
                        .expect("Failed to submit draw command buffer.");

                    let present_info = vk::PresentInfoKHR::default()
                        .wait_semaphores(&signal_semaphores)
                        .swapchains(std::slice::from_ref(&self.base.swapchain_data.swapchain))
                        .image_indices(std::slice::from_ref(&image_index));

                    let queue_present_result = self
                        .base
                        .swapchain_data
                        .swapchain_extension
                        .queue_present(self.base.present_queue, &present_info);

                    match queue_present_result {
                        Ok(_) => (),
                        Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                            return self.base.recreate_swapchain(None)
                        }
                        Err(err) => return Err(Box::new(err)),
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                    return self.base.recreate_swapchain(None)
                }
                Err(err) => return Err(Box::new(err)),
            }

            self.current_frame = (self.current_frame + 1) % self.in_flight_frames;
            Ok(())
        }
    }

    pub fn resize(&mut self, window_size: PhysicalSize<u32>) -> Result<(), Box<dyn Error>> {
        self.base.recreate_swapchain(Some(window_size))
    }
}

impl Drop for Triangle {
    fn drop(&mut self) {
        unsafe {
            let DeviceData { device, .. } = &self.base.device_data;
            let _ = device.device_wait_idle();

            device.destroy_buffer(self.vertex_buffer, None);

            device.free_memory(self.vertex_buffer_memory, None);

            for &fence in &self.in_flight_fences {
                device.destroy_fence(fence, None)
            }

            for &semaphore in &self.render_finished_semaphores {
                device.destroy_semaphore(semaphore, None)
            }

            for &semaphore in &self.image_available_semaphores {
                device.destroy_semaphore(semaphore, None)
            }

            device.destroy_command_pool(self.command_pool, None);

            self.pipelines
                .iter()
                .for_each(|&pipeline| device.destroy_pipeline(pipeline, None));

            device.destroy_pipeline_layout(self.pipeline_layout, None);

            device.destroy_shader_module(self.vert_module, None);

            device.destroy_shader_module(self.frag_module, None);
        }
    }
}
