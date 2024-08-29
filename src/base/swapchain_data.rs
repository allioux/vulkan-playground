use std::error::Error;

use ash::khr::{surface, swapchain};
use ash::{vk, Instance};

use super::DeviceData;

pub struct SwapchainData {
    pub swapchain_extension: swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

impl SwapchainData {
    pub fn new(
        instance: &Instance,
        device_data: &DeviceData,
        &physical_device: &vk::PhysicalDevice,
        surface_extension: &surface::Instance,
        surface: vk::SurfaceKHR,
        surface_resolution: vk::Extent2D,
        surface_format: &vk::SurfaceFormatKHR,
    ) -> Result<SwapchainData, Box<dyn Error>> {
        let device = &device_data.device;
        unsafe {
            let surface_capabilities = surface_extension
                .get_physical_device_surface_capabilities(physical_device, surface)?;

            let desired_image_count = surface_capabilities.min_image_count + 1;

            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };

            let present_modes = surface_extension
                .get_physical_device_surface_present_modes(physical_device, surface)?;

            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_extension = swapchain::Device::new(instance, device);

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

            let swapchain = swapchain_extension.create_swapchain(&swapchain_create_info, None)?;

            let images = swapchain_extension.get_swapchain_images(swapchain)?;

            let image_views = images
                .iter()
                .map(|&image| {
                    device_data.create_image_view(
                        image,
                        surface_format.format,
                        vk::ImageAspectFlags::COLOR,
                    )
                })
                .collect::<Result<_, _>>()?;

            Ok(SwapchainData {
                swapchain_extension,
                swapchain,
                images,
                image_views,
            })
        }
    }
}
