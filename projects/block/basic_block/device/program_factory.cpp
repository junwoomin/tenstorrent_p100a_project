




#include "device_operation.hpp"
#include "host_common.hpp"

namespace ttnn::operations::basic_block {
using namespace tt::tt_metal;

namespace {


enum class ExecutionPath {
    Resident,
    Sharded,
    Stream,
};




ExecutionPath select_execution_path(
    const BasicBlockDeviceOperation::operation_attributes_t& attrs, uint32_t available_cores
) {


    if (attrs.path_override == 0 || attrs.path_override == 1) {
        const auto resident = detail::make_direct_plan(attrs, available_cores);
        if (resident.enabled) {
            return ExecutionPath::Resident;
        }
    }



    if (attrs.path_override == 0 || attrs.path_override == 2) {
        const auto sharded = detail::make_sharded_plan(attrs, available_cores);
        if (sharded.enabled) {
            return ExecutionPath::Sharded;
        }
        TT_FATAL(
            attrs.path_override != 2,
            "Forced BASIC_BLOCK_PATH=sharded cannot fit requested geometry/N shards. "
            "requested_N={} required_l1={} available_l1={}",
            attrs.requested_n_shards,
            sharded.required_bytes,
            sharded.available_bytes
        );
    }

    return ExecutionPath::Stream;
}
}



ProgramDescriptor BasicBlockDeviceOperation::create_descriptor(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensors,
    tensor_return_value_t& outputs
) {
    const auto grid = tensors.x.device()->compute_with_storage_grid_size();
    const auto path = select_execution_path(attrs, grid.x * grid.y);

    switch (path) {
        case ExecutionPath::Resident: {
            return create_resident_descriptor(attrs, tensors, outputs);
        }
        case ExecutionPath::Sharded: {
            return create_sharded_descriptor(attrs, tensors, outputs);
        }
        case ExecutionPath::Stream: {
            return create_stream_descriptor(attrs, tensors, outputs);
        }
    }

    TT_FATAL(false, "Unsupported basic_block execution path");
    return {};
}
}
