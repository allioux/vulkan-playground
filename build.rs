use std::fs;

use shaderc::{self, ShaderKind};

fn shader_kind_to_ext(shader_kind: ShaderKind) -> Option<String> {
    match shader_kind {
        ShaderKind::Vertex => Some("vert".to_string()),
        ShaderKind::Fragment => Some("frag".to_string()),
        _ => None,
    }
}

fn main() {
    let compiler = shaderc::Compiler::new().unwrap();

    for entry in std::fs::read_dir("shaders").unwrap() {
        let entry = entry.unwrap();

        if entry.file_type().unwrap().is_file() {
            let path = entry.path();

            let shader_kind =
                path.extension()
                    .and_then(|ext| match ext.to_string_lossy().as_ref() {
                        "vert" => Some(ShaderKind::Vertex),
                        "frag" => Some(ShaderKind::Fragment),
                        _ => None,
                    });

            match shader_kind {
                Some(shader_kind) => {
                    let source = fs::read_to_string(&path).unwrap();

                    let binary_result = compiler
                        .compile_into_spirv(
                            &source,
                            shader_kind,
                            path.to_str().unwrap(),
                            "main",
                            None,
                        )
                        .unwrap();

                    let path_without_ext = path.with_extension("");
                    let file_name = path_without_ext.file_name().unwrap().to_str().unwrap();
                    let mut new_path = path.parent().unwrap().to_owned();
                    new_path.push(format!(
                        "{}_{}.spv",
                        file_name,
                        shader_kind_to_ext(shader_kind).unwrap()
                    ));

                    fs::write(new_path, binary_result.as_binary_u8()).unwrap();
                }
                None => (),
            }
        }
    }
}
