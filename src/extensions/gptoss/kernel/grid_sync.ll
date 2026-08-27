target triple = "amdgcn-amd-amdhsa"

declare void @__ockl_grid_sync() #0

define i32 @__ockl_grid_sync_i32(i32 %token) #0 {
entry:
  call void @__ockl_grid_sync()
  ret i32 %token
}

attributes #0 = { convergent nounwind }
