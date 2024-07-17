pub fn default<T: Default>() -> T { Default::default() }
pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T=(), E=Error> = std::result::Result<T, E>;
//pub use fehler::throws;
/*use {std::sync::Arc, smallvec::smallvec, vulkano::{device::{Device, Queue}, memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryTypeFilter}, command_buffer::{RecordingCommandBuffer, allocator::StandardCommandBufferAllocator, CopyBufferToImageInfo, BufferImageCopy, CopyImageInfo}, buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer, subbuffer::BufferContents}, image::{Image as GPUImage, ImageCreateInfo, ImageType, ImageUsage, ImageLayout, ImageSubresourceLayers}, format::Format, descriptor_set::allocator::StandardDescriptorSetAllocator, sync::ImageMemoryBarrier}};
use image::{Image, rgba8};

#[throws] pub fn buffer<T: BufferContents>(memory_allocator: Arc<StandardMemoryAllocator>, usage: BufferUsage, len: u32, iter: impl IntoIterator<Item=T>) -> Subbuffer<[T]> {
	let buffer = Buffer::new_slice::<T>(
		memory_allocator.clone(),
		BufferCreateInfo{usage, ..default()},
		AllocationCreateInfo{memory_type_filter: MemoryTypeFilter::PREFER_DEVICE|MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..default()},
		len as u64
	)?;
	{
		let mut write_guard = buffer.write()?;
		for (o, i) in write_guard.iter_mut().zip(iter.into_iter()) { *o = i; }
	}
	buffer
}

pub struct VecGPUImage {
	pub texture_array: Option<Arc<GPUImage>>,
	pub len: usize
}

impl VecGPUImage {
	#[throws] pub fn new(memory_allocator: Arc<StandardMemoryAllocator>, commands: &mut RecordingCommandBuffer, images: impl IntoIterator<Item=Image<&[rgba8]>>) -> Self {
		let images = Box::from_iter(images.into_iter());
		let Some(size) = images.get(0).map(|image| image.size) else { return Ok(Self{texture_array: None, len: 0}) };
		let texture_array = vulkano::image::Image::new(
			memory_allocator.clone(),
			ImageCreateInfo{
				image_type: ImageType::Dim2d,
				format: {assert_eq!(std::mem::size_of::<rgba8>(), 4); Format::R8G8B8A8_SRGB},
				extent: [size.x, size.y, 1],
				array_layers: (images.len() as u32).max(2), // Valid texture arrays starts at len 2 :/
				usage: ImageUsage::TRANSFER_DST|ImageUsage::SAMPLED|ImageUsage::TRANSFER_SRC,
				..default()
			},
			default()
		)?;
		let buffer = Buffer::new_slice::<rgba8>(
			memory_allocator,
			BufferCreateInfo{usage: BufferUsage::TRANSFER_SRC, ..default()},
			AllocationCreateInfo{memory_type_filter: MemoryTypeFilter::PREFER_DEVICE|MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..default()},
			images.len() as u64 * (size.x * size.y) as u64 // * std::mem::size_of::<rgba8>() as u64
		)?;
		{
			let mut write_guard = buffer.write().unwrap();
			for (image, infill) in write_guard.chunks_exact_mut((size.x*size.y) as usize).zip(&*images) {
				image.copy_from_slice(&infill.data);
			}
		}
		commands.copy_buffer_to_image(CopyBufferToImageInfo{
			src_buffer: buffer.into_bytes(),
			dst_image: texture_array.clone(),
			dst_image_layout: ImageLayout::TransferDstOptimal,
			regions: smallvec::smallvec![BufferImageCopy{
				image_subresource: ImageSubresourceLayers{array_layers: 0..images.len() as u32, ..texture_array.subresource_layers()},
				image_extent: texture_array.extent(),
				..default()
			}],
		}).unwrap();
		Self{texture_array: Some(texture_array), len: images.len()}
	}

	#[throws] pub fn push(&mut self, memory_allocator: Arc<StandardMemoryAllocator>, commands: &mut RecordingCommandBuffer, image: Image<&[rgba8]>) {
		let Some(texture_array) = self.texture_array.as_mut() else {
			*self = VecGPUImage::new(memory_allocator, commands, [image])?;
			return Ok(());
		};
		let size = image.size;
		assert_eq!([size.x, size.y, 1], texture_array.extent());
		if self.len == texture_array.array_layers() as usize {
			let next_texture_array = vulkano::image::Image::new(
				memory_allocator.clone(),
				ImageCreateInfo{
					image_type: ImageType::Dim2d,
					format: {assert_eq!(std::mem::size_of::<rgba8>(), 4); Format::R8G8B8A8_SRGB},
					extent: [size.x, size.y, 1],
					array_layers: (self.len as u32*2).max(2), // Valid texture arrays starts at len 2 :/
					usage: ImageUsage::TRANSFER_DST|ImageUsage::SAMPLED|ImageUsage::TRANSFER_SRC,
					..default()
				},
				default()
			)?;
			if self.len > 0 {
				commands.pipeline_barrier(vulkano::sync::DependencyInfo{image_memory_barriers: smallvec![
					ImageMemoryBarrier{
						subresource_range: texture_array.subresource_layers().into(),
						new_layout: ImageLayout::ShaderReadOnlyOptimal, //texture_array.initial_layout(),
						..ImageMemoryBarrier::image(texture_array.clone())
					}],
					..default()
				}).unwrap();
				commands.copy_image(CopyImageInfo::images(texture_array.clone(), next_texture_array.clone()))?;
				commands.pipeline_barrier(vulkano::sync::DependencyInfo{image_memory_barriers: smallvec![
					ImageMemoryBarrier{
						subresource_range: texture_array.subresource_layers().into(),
						new_layout: ImageLayout::ShaderReadOnlyOptimal, //texture_array.initial_layout(),
						..ImageMemoryBarrier::image(texture_array.clone())
					}],
					..default()
				}).unwrap();
			}
			*texture_array = next_texture_array;
		}
		let buffer = Buffer::new_slice::<rgba8>(
			memory_allocator,
			BufferCreateInfo{usage: BufferUsage::TRANSFER_SRC, ..default()},
			AllocationCreateInfo{memory_type_filter: MemoryTypeFilter::PREFER_DEVICE|MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..default()},
			(size.x * size.y) as u64
		)?;
		buffer.write().unwrap().copy_from_slice(&image.data);
		commands.pipeline_barrier(vulkano::sync::DependencyInfo{image_memory_barriers: smallvec![
			ImageMemoryBarrier{
				subresource_range: texture_array.subresource_layers().into(),
				new_layout: ImageLayout::ShaderReadOnlyOptimal, //texture_array.initial_layout(),
				..ImageMemoryBarrier::image(texture_array.clone())
			}],
			..default()
		}).unwrap();
		commands.copy_buffer_to_image(CopyBufferToImageInfo{
			src_buffer: buffer.into_bytes(),
			dst_image: texture_array.clone(),
			dst_image_layout: ImageLayout::TransferDstOptimal,
			regions: smallvec![BufferImageCopy{
				image_subresource: ImageSubresourceLayers{array_layers: self.len as u32..(self.len+1) as u32, ..texture_array.subresource_layers()},
				image_extent: texture_array.extent(),
				..default()
			}],
		}).unwrap();
		commands.pipeline_barrier(vulkano::sync::DependencyInfo{image_memory_barriers: smallvec![
			ImageMemoryBarrier{
				subresource_range: texture_array.subresource_layers().into(),
				new_layout: ImageLayout::ShaderReadOnlyOptimal, //texture_array.initial_layout(),
				..ImageMemoryBarrier::image(texture_array.clone())
			}],
			..default()
		}).unwrap();
		self.len += 1;
	}
}

#[derive(Clone)] pub struct Context {
	pub device: Arc<Device>,
	pub queue: Arc<Queue>,
	pub memory_allocator: Arc<StandardMemoryAllocator>,
	pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
	pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
	pub format: Format,
}*/