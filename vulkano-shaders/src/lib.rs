#![feature(proc_macro_quote)]
struct MacroInput(syn::Ident);
impl syn::parse::Parse for MacroInput { fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> { Ok(Self(input.parse::<syn::Ident>()?)) } }
use proc_macro::{TokenTree, TokenStream, Ident, Literal, Span, quote};
#[proc_macro] pub fn shader(input: TokenStream) -> TokenStream { shader_proc(syn::parse_macro_input!(input as MacroInput)).unwrap() }
#[fehler::throws(Box<dyn std::error::Error>)] fn shader_proc(MacroInput(name): MacroInput) -> TokenStream {
	let spirv = vulkano::shader::spirv::Spirv::new(bytemuck::cast_slice(&std::fs::read(&std::path::Path::new(&std::env::var("OUT_DIR")?).join(name.to_string()+".spv"))?))?;
	TokenStream::from_iter(spirv.types().iter().filter_map(|spirv_struct| {
		use vulkano::shader::spirv::Instruction::*;
		let TypeStruct{result_id: id, member_types} = spirv_struct else { return None; };
		let name_from_id = |id| spirv.id(id).names().iter().find_map(|instruction| if let Name{name, .. } = instruction { Some(name) } else { None });
		let name = name_from_id(*id)?;
		let members = TokenStream::from_iter(member_types.iter().zip(spirv.id(*id).members()).map(|(&id, field)| {
			let member_name = field.names().iter().find_map(|instruction| if let MemberName { name, .. } = instruction { Some(TokenTree::Ident(Ident::new(name, Span::call_site()))) } else { None }).expect("name");
			match spirv.id(id).instruction() {
				TypeFloat{..} => quote!(pub $member_name: half::f16,),
				TypeInt{..} => quote!(pub $member_name: u32,),
				&TypeVector{component_count, component_type, ..} => {
					let component_count = TokenTree::Literal(Literal::usize_unsuffixed(component_count as usize));
					let TypeFloat{width, ..} = spirv.id(component_type).instruction() else {panic!("{component_type:?}")};
					match width {
						16 => quote!(pub $member_name: [half::f16; $component_count],),
						32 => quote!(pub $member_name: [f32; $component_count],),
						_ => unimplemented!()
					}
				},
				TypeMatrix{column_count: column, column_type: id, ..} => {
					let TypeVector{component_count: row, ..} = spirv.id(*id).instruction() else {unimplemented!()};
					let row = match row { 3=>4, 4=>4, _=>unimplemented!() };
					let [row, column] = [row,*column].map(|v| TokenTree::Literal(Literal::usize_unsuffixed(v as usize)));
					quote!(pub $member_name: [[f32; $row]; $column],)
				},
				TypeStruct{..} => {
					let ty = TokenTree::Ident(Ident::new(name_from_id(id).unwrap(), Span::call_site()));
					quote!(pub $member_name: $ty,)
				}
				t => unimplemented!("Unimplemented type conversion (SPV->Rust) {t:?}")
			}
		}));
		let name = TokenTree::Ident(Ident::new(name, Span::call_site()));
		Some(quote!{#[repr(C)]#[derive(Clone,Copy,bytemuck::Zeroable,bytemuck::Pod)] pub struct $name { $members }})
	}))
}
