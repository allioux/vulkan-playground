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

mod graphics;

mod triangle;
use triangle::Triangle;

struct Application {
    window: Option<Arc<Window>>,
    triangle: Option<Triangle>,
}

impl Application {
    fn new() -> Application {
        Application {
            window: None,
            triangle: None,
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

        match Triangle::new(Arc::clone(&window)) {
            Ok(triangle) => {
                self.triangle = Some(triangle);
                self.window = Some(window);
            }
            Err(err) => panic!("Triangle failed with error: {}", err),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
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
            WindowEvent::Resized(size) => match self.triangle {
                Some(ref mut triangle) => {
                    triangle.resize(size);
                    triangle.draw_frame();
                }
                _ => (),
            },
            _ => (),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        match self.triangle {
            Some(ref mut triangle) => {
                triangle.draw_frame();
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
