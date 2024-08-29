use std::error::Error;

use ash::khr::{self};
use ash::prelude::VkResult;
use ash::{vk, Device, Instance};

pub struct DeviceData {
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub surface_extent: vk::Extent2D,
    pub surface_format_khr: vk::SurfaceFormatKHR,
    pub dynamic_rendering_khr: khr::dynamic_rendering::Device,
}

impl DeviceData {
    pub fn new(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        surface_extension: &khr::surface::Instance,
        extent: vk::Extent2D,
    ) -> Result<DeviceData, Box<dyn Error>> {
        unsafe {
            let physical_devices = instance.enumerate_physical_devices()?;

            let (physical_device, queue_family_index) = physical_devices
                .iter()
                .find_map(|&device| {
                    instance
                        .get_physical_device_queue_family_properties(device)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let device_features = instance.get_physical_device_features(device);
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_extension
                                        .get_physical_device_surface_support(
                                            device,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap()
                                    && device_features.sampler_anisotropy == 1;
                            if supports_graphic_and_surface {
                                Some((device, index))
                            } else {
                                None
                            }
                        })
                })
                .ok_or("Could not find a suitable device.")?;

            let enabled_extension_names = [
                khr::swapchain::NAME.as_ptr(),
                khr::multiview::NAME.as_ptr(),
                khr::maintenance2::NAME.as_ptr(),
                khr::create_renderpass2::NAME.as_ptr(),
                khr::depth_stencil_resolve::NAME.as_ptr(),
                khr::dynamic_rendering::NAME.as_ptr(),
                khr::synchronization2::NAME.as_ptr(),
                #[cfg(any(target_os = "macos"))]
                ash::khr::portability_subset::NAME.as_ptr(),
            ];

            let mut dynamic_rendering_feature =
                vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

            let mut synchronization2_feature =
                vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

            let enabled_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index as u32)
                .queue_priorities(&[1.0]);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&enabled_extension_names)
                .enabled_features(&enabled_features)
                .push_next(&mut dynamic_rendering_feature)
                .push_next(&mut synchronization2_feature);

            let device = instance.create_device(physical_device, &device_create_info, None)?;

            let dynamic_rendering_khr = khr::dynamic_rendering::Device::new(&instance, &device);

            let surface_formats =
                surface_extension.get_physical_device_surface_formats(physical_device, surface)?;

            let default_format = surface_formats[0].clone();

            let surface_format_khr: vk::SurfaceFormatKHR = *surface_formats
                .into_iter()
                .filter(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_SRGB
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .collect::<Vec<_>>()
                .first()
                .unwrap_or(&default_format);

            let surface_capabilities = surface_extension
                .get_physical_device_surface_capabilities(physical_device, surface)?;

            let surface_extent = match surface_capabilities.current_extent.width {
                u32::MAX => extent,
                _ => surface_capabilities.current_extent,
            };

            Ok(DeviceData {
                device,
                physical_device,
                queue_family_index: queue_family_index as u32,
                surface_extent,
                surface_format_khr,
                dynamic_rendering_khr,
            })
        }
    }

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
    ) -> VkResult<vk::ImageView> {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);

        unsafe { self.device.create_image_view(&image_view_info, None) }
    }
}
