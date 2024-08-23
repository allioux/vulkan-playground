use std::error::Error;

use ash::khr::surface;
use ash::{vk, Entry, Instance};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

pub struct SurfaceData {
    pub surface_extension: surface::Instance,
    pub surface: vk::SurfaceKHR,
}

impl SurfaceData {
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<SurfaceData, Box<dyn Error>> {
        unsafe {
            let surface_extension = surface::Instance::new(&entry, &instance);

            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .unwrap();

            Ok(SurfaceData {
                surface_extension,
                surface,
            })
        }
    }
}