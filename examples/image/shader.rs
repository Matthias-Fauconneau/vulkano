use {std::sync::Arc, vulkano::{device::Device,	shader::ShaderModule, pipeline::{GraphicsPipeline, graphics::vertex_input::Vertex}}};

pub trait Shader {
	type Vertex: Vertex;
	//const NAME: &'static str;
	fn load(device: Arc<Device>)->Result<Arc<ShaderModule>,vulkano::Validated<vulkano::VulkanError>>;
}

pub struct Pass<S> {
	pub pipeline: Arc<GraphicsPipeline>,
	_marker: std::marker::PhantomData<S>
}

#[macro_export] macro_rules! shader {
	{$name:ident, $vertex:ty, $Name:ident} => {
		pub mod $name {
			use vulkano::{Validated, VulkanError, device::Device, shader::{ShaderModule, ShaderModuleCreateInfo}};
			vulkano_shaders::shader!{$name}
			pub struct Shader;
			use super::*;
			impl $crate::shader::Shader for Shader {
				type Vertex = $vertex;
				fn load(device: Arc<Device>)->Result<Arc<ShaderModule>,Validated<VulkanError>> {
					extern "C" {
						#[link_name=concat!(concat!("_binary_", stringify!($name)), "_spv_start")] static start: [u8; 1];
						#[link_name=concat!(concat!("_binary_", stringify!($name)), "_spv_end")] static end: [u8; 1];
					}
					unsafe{ShaderModule::new(device,
						ShaderModuleCreateInfo::new(&bytemuck::pod_collect_to_vec(std::slice::from_ptr_range(&start..&end))))}
				}
			}
			pub type Pass = $crate::shader::Pass<Shader>;
		}
		pub use $name::Pass as $Name;
	}
}
