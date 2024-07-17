#![feature(slice_from_ptr_range)]
mod vulkan;
mod shader;
use {vulkan::{default, Result}, std::sync::Arc};
use vulkano::{
	device::{physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags, DeviceFeatures},
	buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
	command_buffer::{
		allocator::StandardCommandBufferAllocator, CommandBufferBeginInfo, CommandBufferLevel,
		CommandBufferUsage, CopyBufferToImageInfo, RecordingCommandBuffer, RenderingInfo, RenderingAttachmentInfo
	},
	descriptor_set::{allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet},
	format::Format,
	image::{sampler::{Filter, Sampler, SamplerCreateInfo}, view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
	instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
	memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
	pipeline::{
		graphics::{
			color_blend::ColorBlendState,
			input_assembly::{InputAssemblyState, PrimitiveTopology},
			vertex_input::{Vertex, VertexDefinition},
			viewport::Viewport,
			GraphicsPipelineCreateInfo,
			subpass::PipelineRenderingCreateInfo
		},
		layout::PipelineDescriptorSetLayoutCreateInfo,
		DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
		PipelineShaderStageCreateInfo,
	},
	render_pass::AttachmentStoreOp,
	swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
	sync::{self, GpuFuture},
	DeviceSize, Validated, VulkanError, VulkanLibrary,
};
use winit::{event_loop::EventLoop, window::WindowBuilder,
	event::{Event, WindowEvent::{Resized,RedrawRequested,CloseRequested,KeyboardInput}, KeyEvent, ElementState::Pressed},
	keyboard::{Key::{Named, Character}, NamedKey::Escape}};
use shader::Shader;

#[derive(BufferContents, Vertex)] #[repr(C)] pub struct Position { #[format(R32G32_SFLOAT)] pub position: [f32; 2] }

crate::shader!{quad, Position, Quad}

fn main() -> Result {
	let event_loop = EventLoop::new()?;
	let library = VulkanLibrary::new()?;
	let required_extensions = Surface::required_extensions(&event_loop)?;
	let instance = Instance::new(library, InstanceCreateInfo{flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, enabled_extensions: required_extensions, ..default()})?;
	let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
	let surface = Surface::from_window(instance.clone(), window.clone())?;
	let enabled_extensions = DeviceExtensions{khr_swapchain: true, ..default()};
	let (physical_device, queue_family_index) = instance.enumerate_physical_devices()?.filter(|p| p.supported_extensions().contains(&enabled_extensions)).filter_map(|p|
		p.queue_family_properties().iter().enumerate().position(|(i, q)|
			q.queue_flags.intersects(QueueFlags::GRAPHICS)&& p.surface_support(i as u32, &surface).unwrap_or(false)
		).map(|i| (p, i as u32))
	).min_by_key(|(p, _)| match p.properties().device_type {
		PhysicalDeviceType::DiscreteGpu => 0,
		PhysicalDeviceType::IntegratedGpu => 1,
		PhysicalDeviceType::VirtualGpu => 2,
		PhysicalDeviceType::Cpu => 3,
		PhysicalDeviceType::Other => 4,
		_ => 5,
	}).unwrap();
	let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo{
		enabled_extensions,
		queue_create_infos: vec![QueueCreateInfo{queue_family_index, ..default()}],
		enabled_features: DeviceFeatures{dynamic_rendering: true, ..default()},
		..default()
	})?;
	let queue = queues.next().unwrap();
	let image_format = device.physical_device().surface_formats(&surface, default())?[0].0;
	let (mut swapchain, mut targets) = {
		let surface_capabilities = device.physical_device().surface_capabilities(&surface, default())?;
		Swapchain::new(device.clone(), surface, SwapchainCreateInfo {
			min_image_count: surface_capabilities.min_image_count.max(2),
			image_format,
			image_extent: window.inner_size().into(),
			image_usage: ImageUsage::COLOR_ATTACHMENT,
			..default()
		})?
	};
	let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
	let vertices = [Position{position: [-1./2., -1./2.]}, Position{position: [-1./2., 1./2.]}, Position{position: [1./2., -1./2.]}, Position{position: [1./2., 1./2.]}];
	let vertex_buffer = Buffer::from_iter(memory_allocator.clone(),
		BufferCreateInfo{usage: BufferUsage::VERTEX_BUFFER, ..default()},
		AllocationCreateInfo{memory_type_filter: MemoryTypeFilter::PREFER_DEVICE|MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..default()},
		vertices,
	)?;
	let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), default()));
	let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), default()));
	let mut uploads = RecordingCommandBuffer::new(command_buffer_allocator.clone(), queue.queue_family_index(), CommandBufferLevel::Primary, CommandBufferBeginInfo{usage: CommandBufferUsage::OneTimeSubmit, ..default()})?;
	let texture = {
		let png_bytes = include_bytes!("image_img.png").as_slice();
		let decoder = png::Decoder::new(png_bytes);
		let mut reader = decoder.read_info()?;
		let info = reader.info();
		let extent = [info.width, info.height, 1];
		let upload_buffer = Buffer::new_slice(memory_allocator.clone(), BufferCreateInfo{usage: BufferUsage::TRANSFER_SRC, ..default()}, AllocationCreateInfo{memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..default()},
				(info.width * info.height * 4) as DeviceSize)?;
		reader.next_frame(&mut upload_buffer.write()?)?;
		let image = Image::new(memory_allocator, ImageCreateInfo{image_type: ImageType::Dim2d, format: Format::R8G8B8A8_SRGB, extent,
			usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED, ..default()}, default())?;
		uploads.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(upload_buffer, image.clone()))?;
		ImageView::new_default(image).unwrap()
	};
	let sampler = Sampler::new(device.clone(), SamplerCreateInfo{mag_filter: Filter::Linear, min_filter: Filter::Linear, ..default()})?;
	let pipeline = {
		let shader = quad::Shader::load(device.clone())?;
		let [vertex, fragment] = ["vertex","fragment"].map(|name| PipelineShaderStageCreateInfo::new(shader.entry_point(name).unwrap()));
		let vertex_input_state = (!Position::per_vertex().members.is_empty()).then_some(Position::per_vertex().definition(&vertex.entry_point)?);
		let layout = PipelineLayout::new(device.clone(),
			PipelineDescriptorSetLayoutCreateInfo::from_stages([&vertex, &fragment]).into_pipeline_layout_create_info(device.clone())?)?;
		GraphicsPipeline::new(device.clone(), None, GraphicsPipelineCreateInfo{
			stages: [vertex, fragment].into_iter().collect(),
			vertex_input_state,
			input_assembly_state: Some(InputAssemblyState{topology: PrimitiveTopology::TriangleStrip, ..Default::default()}),
			viewport_state: Some(default()),
			rasterization_state: Some(default()),
			multisample_state: Some(default()),
			color_blend_state: Some(ColorBlendState::with_attachment_states(1, default())),
			dynamic_state: [DynamicState::Viewport].into_iter().collect(),
			subpass: Some(PipelineRenderingCreateInfo{color_attachment_formats: vec![Some(image_format)], ..default()}.into()),
			..GraphicsPipelineCreateInfo::layout(layout)
		})?
	};
	let layout = &pipeline.layout().set_layouts()[0];
	let set = DescriptorSet::new(descriptor_set_allocator, layout.clone(), [WriteDescriptorSet::image_view(0, texture), WriteDescriptorSet::sampler(1, sampler)], [])?;
	let mut recreate_swapchain = false;
	let mut previous_frame_end = Some(uploads.end().unwrap().execute(queue.clone()).unwrap().boxed());
	event_loop.run(move |event, target| (||->Result {
		if let Event::WindowEvent{event, ..} = &event { match event {
			CloseRequested|KeyboardInput{event:KeyEvent{logical_key:Named(Escape), state:Pressed, ..},..} => target.exit(),
			Resized(_) => recreate_swapchain = true,
			RedrawRequested => {
				let image_extent: [u32; 2] = window.inner_size().into();
				if image_extent.contains(&0) { return Ok(()); }
				previous_frame_end.as_mut().unwrap().cleanup_finished();
				if recreate_swapchain {
					(swapchain, targets) = swapchain.recreate(SwapchainCreateInfo{image_extent, ..swapchain.create_info()})?;
					recreate_swapchain = false;
				}
				let (image_index, suboptimal, acquire_future) = match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
					Ok(r) => r,
					Err(VulkanError::OutOfDate) => { recreate_swapchain = true; return Ok(()); }
					Err(e) => panic!("failed to acquire next image: {e}"),
				};
				if suboptimal { recreate_swapchain = true; }
				let mut commands = RecordingCommandBuffer::new(command_buffer_allocator.clone(), queue.queue_family_index(), CommandBufferLevel::Primary, CommandBufferBeginInfo{usage: CommandBufferUsage::OneTimeSubmit, ..default()})?;
				commands
					.begin_rendering(RenderingInfo{
						color_attachments: vec![Some(RenderingAttachmentInfo{store_op: AttachmentStoreOp::Store, ..RenderingAttachmentInfo::image_view(ImageView::new_default(targets[0].clone())?)})],
						..default()
					})?
					.set_viewport(0, [Viewport{extent: image_extent.map(|u32| u32 as f32), ..default()}].into_iter().collect()).unwrap()
					.bind_pipeline_graphics(pipeline.clone())?
					.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline.layout().clone(), 0, set.clone())?
					.bind_vertex_buffers(0, vertex_buffer.clone())?;
				unsafe { commands.draw(vertex_buffer.len() as u32, 1, 0, 0)?; }
				commands.end_rendering()?;
				let command_buffer = commands.end()?;
				let future = previous_frame_end.take().unwrap().join(acquire_future).then_execute(queue.clone(), command_buffer)?.then_swapchain_present(queue.clone(), SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index)).then_signal_fence_and_flush();
				match future.map_err(Validated::unwrap) {
					Ok(future) => previous_frame_end = Some(future.boxed()),
					Err(VulkanError::OutOfDate) => { recreate_swapchain = true; previous_frame_end = Some(sync::now(device.clone()).boxed()); }
					Err(e) => { println!("failed to flush future: {e}"); previous_frame_end = Some(sync::now(device.clone()).boxed()); }
				}
			}
			KeyboardInput{event: KeyEvent{logical_key: Character(key), state: Pressed, ..}, ..} => {
				let key = key.as_str().chars().next().unwrap();
				if key=='q' { target.exit(); }
			}
			_ => (),
		}}
		if recreate_swapchain { window.request_redraw() };
		Ok(())
	})().unwrap())?;
	Ok(())
}
