import ttnn
for path, name in [("transformer","scaled_dot_product_attention_decode"),
                   ("","paged_update_cache"), ("","update_cache")]:
    mod = getattr(ttnn, path) if path else ttnn
    op = getattr(mod, name)
    print("="*30, name, "="*30)
    d = (op.__doc__ or "")[:1600]
    print(d)
