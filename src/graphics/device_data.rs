use std::error::Error;

use ash::khr::{
    create_renderpass2, depth_stencil_resolve, dynamic_rendering,
    maintenance2, multiview, surface, swapchain,
};
use ash::{vk, Device, Instance};

pub struct DeviceData {
    pub device: Device,
    pub dynamic_rendering_extension: dynamic_rendering::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
}

impl DeviceData {
    pub fn new(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        surface_extension: &surface::Instance,
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
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_extension
                                        .get_physical_device_surface_support(
                                            device,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((device, index))
                            } else {
                                None
                            }
                        })
                })
                .ok_or("Could not find a suitable device.")?;

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

            let dynamic_rendering_extension = dynamic_rendering::Device::new(&instance, &device);

            let surface_formats =
                surface_extension.get_physical_device_surface_formats(physical_device, surface)?;

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

            let surface_capabilities = surface_extension
                .get_physical_device_surface_capabilities(physical_device, surface)?;

            let mut desired_image_count = surface_capabilities.min_image_count + 1;

            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }

            let surface_resolution = match surface_capabilities.current_extent.width {
                u32::MAX => extent,
                _ => surface_capabilities.current_extent,
            };

            Ok(DeviceData {
                device,
                dynamic_rendering_extension,
                physical_device,
                queue_family_index: queue_family_index as u32,
                surface_format,
                surface_resolution,
            })
        }
    }
}