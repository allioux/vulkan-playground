use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

mod base;

mod vulkan_example;
use vulkan_example::VulkanExample;

struct Application {
    vulkan_example: Option<VulkanExample>,
    window: Option<Arc<Window>>,
}

impl Application {
    fn new() -> Application {
        Application {
            vulkan_example: None,
            window: None,
        }
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = winit::window::Window::default_attributes()
            .with_title("Vulkan playground")
            .with_inner_size(LogicalSize {
                width: 1024,
                height: 768,
            });

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        window.set_cursor_visible(false);

        match VulkanExample::new(Arc::clone(&window)) {
            Ok(vulkan_example) => {
                self.vulkan_example = Some(vulkan_example);
                self.window = Some(window);
            }
            Err(err) => panic!("VulkanExample failed with error: {}", err),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: event::WindowEvent,
    ) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            }
            | WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => match self.vulkan_example {
                Some(ref mut vulkan_example) => {
                    let _ = vulkan_example.resize(size);
                    let _ = vulkan_example.draw_frame();
                }
                _ => (),
            },
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        match self.vulkan_example {
            Some(ref mut vulkan_example) => {
                let _ = vulkan_example.draw_frame();
            }
            _ => (),
        }
    }
}

fn main() {
    let mut app = Application::new();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let _ = event_loop.run_app(&mut app);
}
