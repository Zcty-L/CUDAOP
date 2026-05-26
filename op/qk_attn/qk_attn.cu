#include "../qk_attn.h"

namespace nvinfer1
{
    __global__ void QKAttnU8Kernel(const uint8_t* q, const uint8_t* k, uint8_t* output, int t, int h, int d, int n)
    {
        int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int t_idx = blockIdx.z;

        if (n_idx >= n || h_idx >= h || t_idx >= t)
            return;

        // Use bitwise logic to track counts across all T steps in parallel.
        // has_one: bit t is 1 if we've seen at least one spike at step t.
        // has_two: bit t is 1 if we've seen at least two spikes at step t.
        uint8_t has_one = 0;
        uint8_t has_two = 0;
        
        int base_idx = t_idx * h * d * n + h_idx * d * n + n_idx;

        for (int d_idx = 0; d_idx < d; ++d_idx)
        {
            uint8_t q_val = q[base_idx + d_idx * n];
            has_two |= (has_one & q_val);
            has_one |= q_val;
        }

        uint8_t attn_mask = has_two;

        for (int d_idx = 0; d_idx < d; ++d_idx)
        {
            output[base_idx + d_idx * n] = k[base_idx + d_idx * n] & attn_mask;
        }
    }

    QKAttnPlugin::QKAttnPlugin(
            int t_, int h_, int d_, int n_, int height_, int width_, nvinfer1::DataType dataType_, int forwardType_)
        : t(t_),
          h(h_),
          d(d_),
          n(n_),
          height(height_),
          width(width_),
          forwardType(forwardType_),
          dataType(dataType_)
    {
        initFieldsToSerialize();
    }

    void QKAttnPlugin::initFieldsToSerialize()
    {
        mDataToSerialize.clear();

        mDataToSerialize.emplace_back(nvinfer1::PluginField("t", &this->t, nvinfer1::PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(nvinfer1::PluginField("h", &this->h, nvinfer1::PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(nvinfer1::PluginField("d", &this->d, nvinfer1::PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(nvinfer1::PluginField("n", &this->n, nvinfer1::PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(
                nvinfer1::PluginField("height", &this->height, nvinfer1::PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(
                nvinfer1::PluginField("width", &this->width, nvinfer1::PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(
                nvinfer1::PluginField("dataType", &this->dataType, nvinfer1::PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(
                nvinfer1::PluginField("forwardType", &this->forwardType, nvinfer1::PluginFieldType::kINT32, 1));

        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
    }

    IPluginV3* QKAttnPlugin::attachToContext(IPluginResourceContext* context) noexcept
    {
        return clone();
    }

    QKAttnPlugin::~QKAttnPlugin() = default;

    IPluginCapability* QKAttnPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
    {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }

    int32_t QKAttnPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out,
                                        int32_t nbOutputs) noexcept
    {
        return 0;
    }

    PluginFieldCollection const* QKAttnPlugin::getFieldsToSerialize() noexcept
    {
        return &mFCToSerialize;
    }

    char const* QKAttnPlugin::getPluginName() const noexcept
    {
        return "QKAttn";
    }

    char const* QKAttnPlugin::getPluginVersion() const noexcept
    {
        return "1";
    }

    char const* QKAttnPlugin::getPluginNamespace() const noexcept
    {
        return mNamespace.c_str();
    }

    void QKAttnPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
    {
        mNamespace = pluginNamespace;
    }

    int32_t QKAttnPlugin::getNbOutputs() const noexcept
    {
        return 1;
    }

    int32_t QKAttnPlugin::getOutputShapes(
            DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
            int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs,
            IExprBuilder &exprBuilder) noexcept
    {
        if (nbInputs < 2 || nbOutputs != 1)
        {
            return -1;
        }

        outputs[0].nbDims = 4;
        outputs[0].d[0] = exprBuilder.constant(t);
        outputs[0].d[1] = exprBuilder.constant(h * d);
        outputs[0].d[2] = exprBuilder.constant(height);
        outputs[0].d[3] = exprBuilder.constant(width);

        return 0;
    }

    int32_t QKAttnPlugin::getOutputDataTypes(
            DataType* outputTypes, int32_t nbOutputs, const DataType* inputTypes,
            int32_t nbInputs) const noexcept
    {
        for (int32_t i = 0; i < nbOutputs; i++)
        {
            outputTypes[i] = this->dataType;
        }
        return 0;
    }

    bool QKAttnPlugin::supportsFormatCombination(
            int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
    {
        return (inOut[pos].desc.type == this->dataType) && (inOut[pos].desc.format == TensorFormat::kLINEAR);
    }

    int32_t QKAttnPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                                          DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
    {
        return 0;
    }

    size_t QKAttnPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                          DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
    {
        return 0;
    }

    IPluginV3* QKAttnPlugin::clone() noexcept
    {
        auto* plugin = new QKAttnPlugin(t, h, d, n, height, width, dataType, forwardType);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    int32_t QKAttnPlugin::enqueue(
            PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
            void const* const * inputs, void* const * outputs, void* workspace,
            cudaStream_t stream) noexcept
    {
        nvinfer1::Dims id = inputDesc[0].dims;
        if (id.nbDims < 4)
        {
            return -1;
        }

        int rt = id.d[0];
        int rh = id.d[1];
        int rd = id.d[2];
        int rn = id.d[3];

        dim3 blockDim(32, 8);
        dim3 gridDim(
                (rn + blockDim.x - 1) / blockDim.x,
                (rh + blockDim.y - 1) / blockDim.y,
                rt);

        auto* q = (const uint8_t*) inputs[0];
        auto* k = (const uint8_t*) inputs[1];
        auto* output = (uint8_t*) outputs[0];

        QKAttnU8Kernel<<<gridDim, blockDim, 0, stream>>>(q, k, output, rt, rh, rd, rn);

        return 0;
    }

    QKAttnPluginCreator::QKAttnPluginCreator()
    {
        mPluginAttributes.clear();

        mPluginAttributes.emplace_back(PluginField("t", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("h", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("d", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("n", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("height", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("width", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("dataType", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("forwardType", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    char const* QKAttnPluginCreator::getPluginName() const noexcept
    {
        return "QKAttn";
    }

    char const* QKAttnPluginCreator::getPluginVersion() const noexcept
    {
        return "1";
    }

    char const* QKAttnPluginCreator::getPluginNamespace() const noexcept
    {
        return mNamespace.c_str();
    }

    void QKAttnPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
    {
        mNamespace = pluginNamespace;
    }

    PluginFieldCollection const* QKAttnPluginCreator::getFieldNames() noexcept
    {
        return &mFC;
    }

    IPluginV3* QKAttnPluginCreator::createPlugin(
            char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
    {
        int t = 0;
        int h = 0;
        int d = 0;
        int n = 0;
        int height = 0;
        int width = 0;
        nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;
        int forwardType = 0;

        const nvinfer1::PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;

            if (!strcmp(attrName, "t"))
            {
                t = *(static_cast<const int32_t*>(fields[i].data));
            }
            else if (!strcmp(attrName, "h"))
            {
                h = *(static_cast<const int32_t*>(fields[i].data));
            }
            else if (!strcmp(attrName, "d"))
            {
                d = *(static_cast<const int32_t*>(fields[i].data));
            }
            else if (!strcmp(attrName, "n"))
            {
                n = *(static_cast<const int32_t*>(fields[i].data));
            }
            else if (!strcmp(attrName, "height"))
            {
                height = *(static_cast<const int32_t*>(fields[i].data));
            }
            else if (!strcmp(attrName, "width"))
            {
                width = *(static_cast<const int32_t*>(fields[i].data));
            }
            else if (!strcmp(attrName, "dataType"))
            {
                dataType = static_cast<nvinfer1::DataType>(*(static_cast<const int32_t*>(fields[i].data)));
            }
            else if (!strcmp(attrName, "forwardType"))
            {
                forwardType = *(static_cast<const int*>(fields[i].data));
            }
        }

        auto* obj = new QKAttnPlugin(t, h, d, n, height, width, dataType, forwardType);
        obj->setPluginNamespace(mNamespace.c_str());

        return obj;
    }
} // namespace nvinfer1
