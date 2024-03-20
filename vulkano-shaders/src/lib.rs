#![feature(proc_macro_quote)]
struct MacroInput(syn::Ident);
impl syn::parse::Parse for MacroInput { fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> { Ok(Self(input.parse::<syn::Ident>()?)) } }
use proc_macro::{TokenTree, TokenStream, Ident, Span, quote};
#[proc_macro] pub fn shader(input: TokenStream) -> TokenStream { shader_proc(syn::parse_macro_input!(input as MacroInput)).unwrap() }
#[fehler::throws(Box<dyn std::error::Error>)] fn shader_proc(MacroInput(name): MacroInput) -> TokenStream {
	let spirv = vulkano::shader::spirv::Spirv::new(bytemuck::cast_slice(&std::fs::read(&std::path::Path::new(&std::env::var("OUT_DIR")?).join(name.to_string()+".spv"))?))?;
	TokenStream::from_iter(spirv.types().iter().filter_map(|spirv_struct| {
		use vulkano::shader::spirv::Instruction;
		let Instruction::TypeStruct{result_id: id, member_types} = spirv_struct else { return None; };
		let name = spirv.id(*id).names().iter().find_map(|instruction| if let Instruction::Name{name, .. } = instruction { Some(name) } else { None })?;
		if name != "Uniforms" { return None; }
		let members = TokenStream::from_iter(member_types.iter().zip(spirv.id(*id).members()).map(|(&id, field)| {
			let name = field.names().iter().find_map(|instruction| if let Instruction::MemberName { name, .. } = instruction { Some(TokenTree::Ident(Ident::new(name, Span::call_site()))) } else { None }).unwrap();
			match spirv.id(id).instruction() {
				Instruction::TypeFloat{..} => quote!(pub $name: f32,),
				Instruction::TypeVector{component_count: 2, ..} => quote!(pub $name: [f32; 2],),
				t => unimplemented!("Unimplemented type conversion (WGSL->Rust) {t:?}")
			}
		}));
		let name = TokenTree::Ident(Ident::new(name, Span::call_site()));
		Some(quote!{#[repr(C)]#[derive(Clone,Copy,bytemuck::Zeroable,bytemuck::Pod)] pub struct $name { $members }})
	}))
}
