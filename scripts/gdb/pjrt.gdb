set breakpoint pending on
#break tt::pjrt::internal::onClientCreate
break tt::pjrt::module_builder::ModuleBuilder::buildModule
commands
    info sharedlibrary pjrt_plugin_tt.so
    info sharedlibrary libTTMLIRRuntime.so
    info sharedlibrary libTTMLIRCompiler.so
end
