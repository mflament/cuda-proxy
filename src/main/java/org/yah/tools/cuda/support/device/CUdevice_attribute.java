package org.yah.tools.cuda.support.device;

@SuppressWarnings("unused")
public enum CUdevice_attribute {
    /**
     * Maximum number of threads per block
     **/
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK(1),
    /**
     * Maximum block dimension X
     **/
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X(2),
    /**
     * Maximum block dimension Y
     **/
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y(3),
    /**
     * Maximum block dimension Z
     **/
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z(4),
    /**
     * Maximum grid dimension X
     **/
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X(5),
    /**
     * Maximum grid dimension Y
     **/
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y(6),
    /**
     * Maximum grid dimension Z
     **/
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z(7),
    /**
     * Maximum shared memory available per block in bytes
     **/
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK(8),
    /**
     * Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
     **/
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK(8),
    /**
     * Memory available on device for __constant__ variables in a CUDA C kernel in bytes
     **/
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY(9),
    /**
     * Warp size in threads
     **/
    CU_DEVICE_ATTRIBUTE_WARP_SIZE(10),
    /**
     * Maximum pitch in bytes allowed by memory copies
     **/
    CU_DEVICE_ATTRIBUTE_MAX_PITCH(11),
    /**
     * Maximum number of 32-bit registers available per block
     **/
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK(12),
    /**
     * Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
     **/
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK(12),
    /**
     * Typical clock frequency in kilohertz
     **/
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE(13),
    /**
     * Alignment requirement for textures
     **/
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT(14),
    /**
     * Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.
     **/
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP(15),
    /**
     * Number of multiprocessors on device
     **/
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT(16),
    /**
     * Specifies whether there is a run time limit on kernels
     **/
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT(17),
    /**
     * Device is integrated with host memory
     **/
    CU_DEVICE_ATTRIBUTE_INTEGRATED(18),
    /**
     * Device can map host memory into CUDA address space
     **/
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY(19),
    /**
     * Compute mode (See CUcomputemode for details)
     **/
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE(20),
    /**
     * Maximum 1D texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH(21),
    /**
     * Maximum 2D texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH(22),
    /**
     * Maximum 2D texture height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT(23),
    /**
     * Maximum 3D texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH(24),
    /**
     * Maximum 3D texture height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT(25),
    /**
     * Maximum 3D texture depth
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH(26),
    /**
     * Maximum 2D layered texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH(27),
    /**
     * Maximum 2D layered texture height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT(28),
    /**
     * Maximum layers in a 2D layered texture
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS(29),
    /**
     * Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH(27),
    /**
     * Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT(28),
    /**
     * Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES(29),
    /**
     * Alignment requirement for surfaces
     **/
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT(30),
    /**
     * Device can possibly execute multiple kernels concurrently
     **/
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS(31),
    /**
     * Device has ECC support enabled
     **/
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED(32),
    /**
     * PCI bus ID of the device
     **/
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID(33),
    /**
     * PCI device ID of the device
     **/
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID(34),
    /**
     * Device is using TCC driver model
     **/
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER(35),
    /**
     * Peak memory clock frequency in kilohertz
     **/
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE(36),
    /**
     * Global memory bus width in bits
     **/
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH(37),
    /**
     * Size of L2 cache in bytes
     **/
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE(38),
    /**
     * Maximum resident threads per multiprocessor
     **/
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR(39),
    /**
     * Number of asynchronous engines
     **/
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT(40),
    /**
     * Device shares a unified address space with the host
     **/
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING(41),
    /**
     * Maximum 1D layered texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH(42),
    /**
     * Maximum layers in a 1D layered texture
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS(43),
    /**
     * Deprecated, do not use.
     **/
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER(44),
    /**
     * Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH(45),
    /**
     * Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT(46),
    /**
     * Alternate maximum 3D texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE(47),
    /**
     * Alternate maximum 3D texture height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE(48),
    /**
     * Alternate maximum 3D texture depth
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE(49),
    /**
     * PCI domain ID of the device
     **/
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID(50),
    /**
     * Pitch alignment requirement for textures
     **/
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT(51),
    /**
     * Maximum cubemap texture width/height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH(52),
    /**
     * Maximum cubemap layered texture width/height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH(53),
    /**
     * Maximum layers in a cubemap layered texture
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS(54),
    /**
     * Maximum 1D surface width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH(55),
    /**
     * Maximum 2D surface width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH(56),
    /**
     * Maximum 2D surface height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT(57),
    /**
     * Maximum 3D surface width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH(58),
    /**
     * Maximum 3D surface height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT(59),
    /**
     * Maximum 3D surface depth
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH(60),
    /**
     * Maximum 1D layered surface width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH(61),
    /**
     * Maximum layers in a 1D layered surface
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS(62),
    /**
     * Maximum 2D layered surface width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH(63),
    /**
     * Maximum 2D layered surface height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT(64),
    /**
     * Maximum layers in a 2D layered surface
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS(65),
    /**
     * Maximum cubemap surface width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH(66),
    /**
     * Maximum cubemap layered surface width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH(67),
    /**
     * Maximum layers in a cubemap layered surface
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS(68),
    /**
     * Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH(69),
    /**
     * Maximum 2D linear texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH(70),
    /**
     * Maximum 2D linear texture height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT(71),
    /**
     * Maximum 2D linear texture pitch in bytes
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH(72),
    /**
     * Maximum mipmapped 2D texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH(73),
    /**
     * Maximum mipmapped 2D texture height
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT(74),
    /**
     * Major compute capability version number
     **/
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR(75),
    /**
     * Minor compute capability version number
     **/
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR(76),
    /**
     * Maximum mipmapped 1D texture width
     **/
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH(77),
    /**
     * Device supports stream priorities
     **/
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED(78),
    /**
     * Device supports caching globals in L1
     **/
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED(79),
    /**
     * Device supports caching locals in L1
     **/
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED(80),
    /**
     * Maximum shared memory available per multiprocessor in bytes
     **/
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR(81),
    /**
     * Maximum number of 32-bit registers available per multiprocessor
     **/
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR(82),
    /**
     * Device can allocate managed memory on this system
     **/
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY(83),
    /**
     * Device is on a multi-GPU board
     **/
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD(84),
    /**
     * Unique id for a group of devices on the same multi-GPU board
     **/
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID(85),
    /**
     * Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
     **/
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED(86),
    /**
     * Ratio of single precision performance (in floating-point operations per second) to double precision performance
     **/
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO(87),
    /**
     * Device supports coherently accessing pageable memory without calling cudaHostRegister on it
     **/
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS(88),
    /**
     * Device can coherently access managed memory concurrently with the CPU
     **/
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS(89),
    /**
     * Device supports compute preemption.
     **/
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED(90),
    /**
     * Device can access host registered memory at the same virtual address as the CPU
     **/
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM(91),
    /**
     * Deprecated, along with v1 MemOps API, cuStreamBatchMemOp and related APIs are supported.
     **/
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1(92),
    /**
     * Deprecated, along with v1 MemOps API, 64-bit operations are supported in cuStreamBatchMemOp and related APIs.
     **/
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1(93),
    /**
     * Deprecated, along with v1 MemOps API, CU_STREAM_WAIT_VALUE_NOR is supported.
     **/
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1(94),
    /**
     * Device supports launching cooperative kernels via cuLaunchCooperativeKernel
     **/
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH(95),
    /**
     * Deprecated, cuLaunchCooperativeKernelMultiDevice is deprecated.
     **/
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH(96),
    /**
     * Maximum optin shared memory per block
     **/
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN(97),
    /**
     * The CU_STREAM_WAIT_VALUE_FLUSH flag and the CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See Stream Memory Operations for additional details.
     **/
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES(98),
    /**
     * Device supports host memory registration via cudaHostRegister.
     **/
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED(99),
    /**
     * Device accesses pageable memory via the host's page tables.
     **/
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES(100),
    /**
     * The host can directly access managed memory on the device without migration.
     **/
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST(101),
    /**
     * Deprecated, Use CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
     **/
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED(102),
    /**
     * Device supports virtual memory management APIs like cuMemAddressReserve, cuMemCreate, cuMemMap and related APIs
     **/
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED(102),
    /**
     * Device supports exporting memory to a posix file descriptor with cuMemExportToShareableHandle, if requested via cuMemCreate
     **/
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED(103),
    /**
     * Device supports exporting memory to a Win32 NT handle with cuMemExportToShareableHandle, if requested via cuMemCreate
     **/
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED(104),
    /**
     * Device supports exporting memory to a Win32 KMT handle with cuMemExportToShareableHandle, if requested via cuMemCreate
     **/
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED(105),
    /**
     * Maximum number of blocks per multiprocessor
     **/
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR(106),
    /**
     * Device supports compression of memory
     **/
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED(107),
    /**
     * Maximum L2 persisting lines capacity setting in bytes.
     **/
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE(108),
    /**
     * Maximum value of CUaccessPolicyWindow::num_bytes.
     **/
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE(109),
    /**
     * Device supports specifying the GPUDirect RDMA flag with cuMemCreate
     **/
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED(110),
    /**
     * Shared memory reserved by CUDA driver per block in bytes
     **/
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK(111),
    /**
     * Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays
     **/
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED(112),
    /**
     * Device supports using the cuMemHostRegister flag CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU
     **/
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED(113),
    /**
     * External timeline semaphore interop is supported on the device
     **/
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED(114),
    /**
     * Device supports using the cuMemAllocAsync and cuMemPool family of APIs
     **/
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED(115),
    /**
     * Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see <a href="https://docs.nvidia.com/cuda/gpudirect-rdma">doc</a> for more information)
     **/
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED(116),
    /**
     * The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the CUflushGPUDirectRDMAWritesOptions enum
     **/
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS(117),
    /**
     * GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See CUGPUDirectRDMAWritesOrdering for the numerical values returned here.
     **/
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING(118),
    /**
     * Handle types supported with mempool based IPC
     **/
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES(119),
    /**
     * Indicates device supports cluster launch
     **/
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH(120),
    /**
     * Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
     **/
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED(121),
    /**
     * 64-bit operations are supported in cuStreamBatchMemOp and related MemOp APIs.
     **/
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS(122),
    /**
     * CU_STREAM_WAIT_VALUE_NOR is supported by MemOp APIs.
     **/
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR(123),
    /**
     * Device supports buffer sharing with dma_buf mechanism.
     **/
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED(124),
    /**
     * Device supports IPC Events.
     **/
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED(125),
    /**
     * Number of memory domains the device supports.
     **/
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT(126),
    /**
     * Device supports accessing memory using Tensor Map.
     **/
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED(127),
    /**
     * Device supports exporting memory to a fabric handle with cuMemExportToShareableHandle() or requested with cuMemCreate()
     **/
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED(128),
    /**
     * Device supports unified function pointers.
     **/
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS(129),
    /**
     * CU_DEVICE_ATTRIBUTE_NUMA_ID = 131
     **/
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG(130),
    /**
     * Device supports switch multicast and reduction operations.
     **/
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED(132),
    /**
     * Indicates if contexts created on this device will be shared via MPS
     **/
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED(133),
    /**
     * NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA.
     **/
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID(134),

    CU_DEVICE_ATTRIBUTE_MAX(135);

    private final int value;

    CUdevice_attribute(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }
}
