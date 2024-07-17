@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var linear_interpolation: sampler;

struct Vertex {
 @location(0) position: vec2<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(1) texture_coordinates: vec2<f32>,
}

@vertex fn vertex(vertex: Vertex) -> VertexOutput {
	return VertexOutput(vec4(vertex.position, 0., 1.), vertex.position+1./2.);
}

@fragment fn fragment(vertex: VertexOutput) -> @location(0) vec4<f32> {
	return textureSample(image, linear_interpolation, vec2(vertex.texture_coordinates.x, vertex.texture_coordinates.y));
}
