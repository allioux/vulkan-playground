use std::error::Error;
use std::mem::offset_of;
use std::sync::Arc;
use std::time::Instant;

use ash::prelude::VkResult;
use ash::util::{read_spv, Align};
use ash::vk::{self};
use image::{EncodableLayout, GenericImageView, ImageReader};
use nalgebra_glm::{vec3, Mat4, Vec3};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::base::{
    Base, DeviceData, ImageDescriptor, PipelineDescriptor, TransitionImageLayoutDesc,
};

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

#[derive(Clone, Copy)]
#[repr(C)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
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
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<Align<UniformBufferObject>>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    instant: Instant,
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
                pos: [-0.5, -0.5],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                pos: [0.5, -0.5],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                pos: [0.5, 0.5],
                color: [0.0, 0.0, 1.0],
            },
            Vertex {
                pos: [-0.5, 0.5],
                color: [1.0, 1.0, 1.0],
            },
        ];

        let indices: Vec<u16> = vec![0, 1, 2, 2, 3, 0];

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

            // UNIFORM BUFFERS

            let buffer_size = size_of::<UniformBufferObject>() as u64;
            let mut uniform_buffers = vec![];
            let mut uniform_buffers_memory = vec![];
            let mut uniform_buffers_mapped: Vec<Align<UniformBufferObject>> = vec![];
            for _ in 0..in_flight_frames {
                let (buffer, buffer_memory, memory_requirements) = base.create_buffer(
                    buffer_size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                uniform_buffers.push(buffer);
                uniform_buffers_memory.push(buffer_memory);
                let ptr = base.device_data.device.map_memory(
                    buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )?;
                uniform_buffers_mapped.push(Align::new(
                    ptr,
                    align_of::<UniformBufferObject>() as u64,
                    memory_requirements.size,
                ));
            }

            // DESCRIPTOR POOL

            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(in_flight_frames);

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&pool_size))
                .max_sets(in_flight_frames);

            let descriptor_pool = device.create_descriptor_pool(&pool_info, None)?;

            // DESCRIPTOR SET LAYOUT

            let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX);

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(std::slice::from_ref(&ubo_layout_binding));

            let descriptor_set_layout = base
                .device_data
                .device
                .create_descriptor_set_layout(&layout_info, None)?;

            // DESCRIPTOR SETS

            let layouts = vec![descriptor_set_layout; in_flight_frames as usize];
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);

            let descriptor_sets = device.allocate_descriptor_sets(&descriptor_set_allocate_info)?;

            for i in 0..in_flight_frames {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffers[i as usize])
                    .offset(0)
                    .range(size_of::<UniformBufferObject>() as u64);

                let descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_sets[i as usize])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(std::slice::from_ref(&buffer_info));

                device.update_descriptor_sets(&[descriptor_write], &[]);
            }

            // PIPELINE

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

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

            // TEXTURE IMAGE

            let image = ImageReader::open("textures/statue.jpg")?.decode()?;
            let image_data = image.to_rgba8();
            let (width, height) = image.dimensions();
            let image_extent = vk::Extent3D {
                height,
                width,
                depth: 1,
            };
            let buffer_size = (height * width * 4 * size_of::<u8>() as u32) as u64;

            let (staging_buffer, staging_buffer_memory, _) = base.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let ptr = device.map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )?;

            let mut image_slice = Align::new(ptr, align_of::<u8>() as u64, buffer_size);
            image_slice.copy_from_slice(image_data.as_bytes());
            device.unmap_memory(staging_buffer_memory);

            let (image, image_memory, _) = base.create_image(ImageDescriptor {
                extent: image_extent,
                format: vk::Format::R8G8B8A8_SRGB,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })?;

            let command_buffer = base.begin_single_time_commands(command_pool)?;
            base.transition_image_layout(
                command_buffer,
                image,
                TransitionImageLayoutDesc::from_layouts(
                    vk::Format::R8G8B8A8_SRGB,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                )?,
            );
            base.end_single_time_commands(command_pool, command_buffer)?;

            base.copy_buffer_to_image(command_pool, staging_buffer, image, image_extent)?;

            let command_buffer = base.begin_single_time_commands(command_pool)?;
            base.transition_image_layout(
                command_buffer,
                image,
                TransitionImageLayoutDesc::from_layouts(
                    vk::Format::R8G8B8A8_SRGB,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::READ_ONLY_OPTIMAL,
                )?,
            );
            base.end_single_time_commands(command_pool, command_buffer)?;

            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);

            // IMAGE VIEW

            let image_view = base
                .device_data
                .create_image_view(image, vk::Format::R8G8B8A8_SRGB)?;

            // VERTEX & INDEX BUFFERS

            let (vertex_buffer, vertex_buffer_memory) = base.create_and_populate_buffer(
                command_pool,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vertices,
            )?;

            let (index_buffer, index_buffer_memory) = base.create_and_populate_buffer(
                command_pool,
                vk::BufferUsageFlags::INDEX_BUFFER,
                indices,
            )?;

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
                index_buffer,
                index_buffer_memory,
                uniform_buffers,
                uniform_buffers_memory,
                uniform_buffers_mapped,
                descriptor_pool,
                descriptor_set_layout,
                descriptor_sets,
                image,
                image_memory,
                image_view,
                instant: Instant::now(),
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
                            self.pipeline_layout,
                            &[self.vertex_buffer],
                            self.index_buffer,
                            vk::IndexType::UINT16,
                            6,
                            &self.descriptor_sets,
                            frame_index as u32,
                        )
                        .expect("Failed to record the command buffer.");

                    // UPDATING UNIFORM DATA
                    let elapsed = self.instant.elapsed();

                    let model_matrix = nalgebra_glm::rotate(
                        &Mat4::identity(),
                        90.0_f32.to_radians() * elapsed.as_millis() as f32 / 1000.,
                        &Vec3::z(),
                    );

                    let view_matrix = nalgebra_glm::look_at(
                        &vec3(2.0, 2.0, 2.0),
                        &vec3(0.0, 0.0, 0.0),
                        &vec3(0.0, 0.0, 1.0),
                    );

                    let proj_matrix = nalgebra_glm::perspective(
                        self.base.device_data.surface_extent.width as f32
                            / self.base.device_data.surface_extent.height as f32,
                        45.0_f32.to_radians(),
                        0.1,
                        10.0,
                    );

                    let mut ubo = UniformBufferObject {
                        model: model_matrix,
                        view: view_matrix,
                        proj: proj_matrix,
                    };

                    ubo.proj[(1, 1)] *= -1.;

                    self.uniform_buffers_mapped[frame_index].copy_from_slice(&[ubo]);

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

            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_buffer_memory, None);

            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);

            for &buffer in &self.uniform_buffers {
                device.destroy_buffer(buffer, None);
            }

            for &buffer_memory in &self.uniform_buffers_memory {
                device.free_memory(buffer_memory, None);
            }

            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.image_memory, None);

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
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.vert_module, None);
            device.destroy_shader_module(self.frag_module, None);
        }
    }
}
