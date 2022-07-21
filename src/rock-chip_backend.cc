#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

#include "rock-chip_backend.h"

namespace triton { namespace backend{namespace rockchip{

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Name of the input and output tensor
  const std::string& InputTensorName() const { return input_name_; }
  const std::vector<std::string>& OutputTensorName() const { return output_name_; }

  // Datatype of the input and output tensor
  TRITONSERVER_DataType TensorDataType() const { return datatype_; }
  TRITONSERVER_DataType OutputTensorDataType(std::string key) const { 
    auto it = output_dt_.find(key);
    if(it!=output_dt_.end())
      return it->second;
    return TRITONSERVER_TYPE_UINT8;
   }
  std::vector<int64_t>& getOutputshapes(std::string outputname){
    auto it = output_shape_.find(outputname);
    if(it!=output_shape_.end())
      return it->second;
    return it->second;
  }
  // Shape of the input and output tensor as given in the model
  // configuration file. This shape will not include the batch
  // dimension (if the model has one).
  const std::vector<int64_t>& TensorNonBatchShape() const { return nb_shape_; }

  // Shape of the input and output tensor, including the batch
  // dimension (if the model has one). This method cannot be called
  // until the model is completely loaded and initialized, including
  // all instances of the model. In practice, this means that backend
  // should only call it in TRITONBACKEND_ModelInstanceExecute.
  TRITONSERVER_Error* TensorShape(std::vector<int64_t>& shape);

  // Validate that this model is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  std::string input_name_;
  // std::string output_name_;
  std::vector<std::string> output_name_;

  TRITONSERVER_DataType datatype_;
  std::map<std::string,TRITONSERVER_DataType>output_dt_;
  std::map<std::string, std::vector<int64_t>>output_shape_;
  bool shape_initialized_;
  std::vector<int64_t> nb_shape_;
  std::vector<int64_t> shape_;
};

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), shape_initialized_(false)
{
  // Validate that the model's configuration matches what is supported
  // by this backend.
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));
  // ModelState* x=reinterpret_cast<ModelState*>(backend);
  
  // LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("bbbbbbbbbbbbbbbbbbb")+std::string("x->batch_output_map_.size(); ")+std::to_string(x->batch_output_map_.size())).c_str());
  // // common::TritonJson::WriteBuffer buffer;
  // //   THROW_IF_BACKEND_MODEL_ERROR(ModelConfig().PrettyWrite(&buffer));
  // //   LOG_MESSAGE(
  // //       // TRITONSERVER_LOG_VERBOSE,
  // //       TRITONSERVER_LOG_INFO,
  // //       (std::string("model configuration:\n") + buffer.Contents()).c_str());
  
  // LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("aaaaaaaaaaaaaaaaaaaa")+std::string("x->batch_output_map_.size(); ")+std::to_string(x->batch_output_map_.size())).c_str());
  // // THROW_IF_BACKEND_MODEL_ERROR(
  // //     BatchOutput::ParseFromModelConfig(ModelConfig(), &batch_outputs_));
  // {
  //   batch_outputs_.clear();

  // }
  // LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("batch_outputs_")+std::to_string(batch_outputs_.size())).c_str());
  // for (const auto& batch_output : batch_outputs_) {
  //   for (const auto& name : batch_output.TargetNames()) {
  //     batch_output_map_.emplace(name, &batch_output);
  //   }
  // }
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::TensorShape(std::vector<int64_t>& shape)
{
  // This backend supports models that batch along the first dimension
  // and those that don't batch. For non-batch models the output shape
  // will be the shape from the model configuration. For batch models
  // the output shape will be the shape from the model configuration
  // prepended with [ -1 ] to represent the batch dimension. The
  // backend "responder" utility used below will set the appropriate
  // batch dimension value for each response. The shape needs to be
  // initialized lazily because the SupportsFirstDimBatching function
  // cannot be used until the model is completely loaded.
  if (!shape_initialized_) {
    bool supports_first_dim_batching;
    RETURN_IF_ERROR(SupportsFirstDimBatching(&supports_first_dim_batching));
    if (supports_first_dim_batching) {
      shape_.push_back(-1);
    }

    shape_.insert(shape_.end(), nb_shape_.begin(), nb_shape_.end());
    shape_initialized_ = true;
  }

  shape = shape_;

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // If verbose logging is enabled, dump the model's configuration as
  // JSON into the console output.
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        // TRITONSERVER_LOG_VERBOSE,
        TRITONSERVER_LOG_INFO,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
  }

  // ModelConfig is the model configuration as a TritonJson
  // object. Use the TritonJson utilities to parse the JSON and
  // determine if the configuration is supported by this backend.
  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

  // The model must have exactly 1 input and 1 output.
//   RETURN_ERROR_IF_FALSE(
//       inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
//       std::string("model configuration must have 1 input"));
//   RETURN_ERROR_IF_FALSE(
//       outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
//       std::string("model configuration must have 1 output"));

  common::TritonJson::Value input, output;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  // RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

  // Record the input and output name in the model state.
  const char* input_name;
  size_t input_name_len;
  RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
  input_name_ = std::string(input_name);

  for(size_t i=0;i<outputs.ArraySize();i++){
    const char* output_name;
    size_t output_name_len;
    RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));
    RETURN_IF_ERROR(
        output.MemberAsString("name", &output_name, &output_name_len));
        output_name_.push_back(std::string(output_name));
    
    const char* dt_name;
    size_t dt_name_len;
    RETURN_IF_ERROR(output.MemberAsString("data_type", &dt_name, &dt_name_len));
    output_dt_.insert(std::make_pair(std::string(output_name),getTritonDT(std::string(dt_name))));


    common::TritonJson::Value model_config_dims;
    RETURN_IF_ERROR(output.MemberAsArray("dims", &model_config_dims));
    std::vector<int64_t> dim_vec;
    
    RETURN_IF_ERROR(DimsJsonToDimVec(model_config_dims, &dim_vec));
    output_shape_.insert(std::make_pair(std::string(output_name),dim_vec));

    
  }
    // {
    //   std::stringstream ss;
    // auto&xx=getOutputshapes("output");
    // for (auto it = xx.begin(); it != xx.end(); it++)    {
    //     if (it != xx.begin()) {
    //         ss << " ";
    //     }
    //     ss << *it;
    // }
    // std::cout << ss.str() << std::endl;  
    // std::cout << "line 224: --------------------------------------------"<<std::endl;    
    // ss.str("");
    // xx=getOutputshapes("376");
    // for (auto it = xx.begin(); it != xx.end(); it++)    {
    //     if (it != xx.begin()) {
    //         ss << " ";
    //     }
    //     ss << *it;
    // }
    // std::cout << ss.str() << std::endl;  
    // std::cout << "line 224: --------------------------------------------"<<std::endl;    
    // ss.str("");
    // xx=getOutputshapes("377");
    // for (auto it = xx.begin(); it != xx.end(); it++)    {
    //     if (it != xx.begin()) {
    //         ss << " ";
    //     }
    //     ss << *it;
    // }
    // std::cout << ss.str() << std::endl;  
    // std::cout << "line 224: --------------------------------------------"<<std::endl;    
    // }
  // Input and output must have same datatype
  std::string input_dtype, output_dtype;
  RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
//   RETURN_ERROR_IF_FALSE(
//       input_dtype == output_dtype, TRITONSERVER_ERROR_INVALID_ARG,
//       std::string("expected input and output datatype to match, got ") +
//           input_dtype + " and " + output_dtype);
  datatype_ = ModelConfigDataTypeToTritonServerDataType(input_dtype);

  // Input and output must have same shape. Reshape is not supported
  // on either input or output so flag an error is the model
  // configuration uses it.
//   triton::common::TritonJson::Value reshape;
//   RETURN_ERROR_IF_TRUE(
//       input.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
//       std::string("reshape not supported for input tensor"));
//   RETURN_ERROR_IF_TRUE(
//       output.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
//       std::string("reshape not supported for output tensor"));

  std::vector<int64_t> input_shape, output_shape;
  RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

//   RETURN_ERROR_IF_FALSE(
//       input_shape == output_shape, TRITONSERVER_ERROR_INVALID_ARG,
//       std::string("expected input and output shape to match, got ") +
//           backend::ShapeToString(input_shape) + " and " +
//           backend::ShapeToString(output_shape));

  nb_shape_ = input_shape;

//   std::string input_string(input_shape.begin(),input_shape.end()),output_string(output_shape.begin(),output_shape.end());
  std::ostringstream oss;

  if (!input_shape.empty() && !output_shape.empty())
  {
    // Convert all but the last element to avoid a trailing ","
    std::copy(input_shape.begin(), input_shape.end()-1,
        std::ostream_iterator<int>(oss, ","));

    // Now add the last element with no delimiter
    oss << input_shape.back();
    oss<< "][";
    // Convert all but the last element to avoid a trailing ","
    std::copy(output_shape.begin(), output_shape.end()-1,
        std::ostream_iterator<int>(oss, ","));

    // Now add the last element with no delimiter
    oss << output_shape.back();
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("validated model config for input and output shape is:[")+oss.str()+std::string("]")).c_str());
  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }
  rknn_context* getRknnContext(){return &ctx;}
  
  // The maximum possible size of the TensorRT tensor and the
  // corresponding allocated GPU buffer across all optimization
  // profile.
  using BatchInputData = std::pair<BatchInput, std::unique_ptr<BackendMemory>>;
  std::vector<std::pair<std::string, std::int64_t>> outputs_bytes;
  struct IOBindingInfo {
    IOBindingInfo()
        : byte_size_(0), buffer_(nullptr), device_buffer_(nullptr),
          memory_type_(TRITONSERVER_MEMORY_GPU), memory_type_id_(0),
          buffer_is_ragged_(false), is_linear_format_(true),
          vectorized_dim_(-1), components_per_element_(1),
          is_state_output_(false), is_requested_output_tensor_(false)
    {
    }
    uint64_t byte_size_;
    void* buffer_;
    void* device_buffer_;
    TRITONSERVER_MemoryType memory_type_;
    int64_t memory_type_id_;
    bool buffer_is_ragged_;
    bool is_linear_format_;
    int vectorized_dim_;
    int components_per_element_;
    const BatchOutput* batch_output_;
    // Instructions on constructing the batch input and the CPU buffer
    // for storing mutable data
    std::shared_ptr<BatchInputData> batch_input_;
    // Store the pair of input name to look up and output shape
    // for output scattering
    std::pair<std::string, std::vector<int64_t>> io_shape_mapping_;

    // Indicates whether the output is a state output.
    bool is_state_output_;

    // Indicates whether the output is a output tensor.
    bool is_requested_output_tensor_;
  };
  TRITONSERVER_Error* InitIOBindingBuffers(); //assume input num always 1
  // There are Context::num_expected_bindings_ number of IOBindingInfo
  // elements for copy stream.
  std::vector<IOBindingInfo> io_binding_infos_;
 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
  {
    deviceArch=std::move(std::string(getBuild()));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("rk backends running on device arch :")+deviceArch).c_str());
  }
  TRITONSERVER_Error* InitializeConfigShapeOutputBindings(
      common::TritonJson::Value& config_output);
  ModelState* model_state_;
  rknn_context ctx;
  std::string deviceArch{};
  unsigned char *model=NULL; // useless
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state){
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
    auto* ctx =(*state)->getRknnContext();auto myself=*state;
    TRITONBACKEND_ArtifactType artifatct_type; 
    const char *path = ""; 
    int ret = -1;
    rknn_mem_size memSize{};rknn_sdk_version rknnSdkVersion{};
    RETURN_IF_ERROR(TRITONBACKEND_ModelRepository((*state)->model_state_->TritonModel(), &artifatct_type, &path)); 
    
    std::stringstream ss;
    ss<<path<<'/'<<(*state)->model_state_->Version()<<"/model.rknn";

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("rk backend will load model from :")+ss.str()).c_str());
     // int model_len;
     // (*state)->model = load_model(ss.str().c_str(),&model_len);
     
     //  ret = rknn_init(&((*state)->ctx), (*state)->model, 0, 0,0);
     ret = rknn_init(ctx,(void*)ss.str().c_str(),0,0,0);
     if(ret < 0)
       LOG_MESSAGE(TRITONSERVER_LOG_ERROR,(std::string("rknn_init fail! ret= :")+std::to_string(ret)).c_str());
     else
       LOG_MESSAGE(TRITONSERVER_LOG_ERROR,(std::string("rknn_init succeed! ret= :")+std::to_string(ret)).c_str());
     ret=rknn_query(*ctx,RKNN_QUERY_SDK_VERSION,(void*)&rknnSdkVersion,sizeof(rknnSdkVersion));
     LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("rknn sdk api version: ")+std::string(rknnSdkVersion.api_version)+
       std::string(", rknn driver version: ")+std::string(rknnSdkVersion.drv_version)).c_str());
     if(!myself->deviceArch.compare("ARM64")){
      ret = rknn_query(*ctx, RKNN_QUERY_MEM_SIZE, &memSize, sizeof(memSize));
      LOG_MESSAGE(TRITONSERVER_LOG_INFO,(
            std::string("\n rknn_mem_size : \n\t total_weight_size : ")+
            std::to_string(memSize.total_weight_size)+std::string("\n\t total_internal_size : ")+
            std::to_string(memSize.total_internal_size)).c_str());
     }
     RETURN_IF_ERROR((*state)->InitIOBindingBuffers());
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::InitializeConfigShapeOutputBindings(
    common::TritonJson::Value& config_output){
          // todo sth.
          return nullptr;
    }


TRITONSERVER_Error*
ModelInstanceState::InitIOBindingBuffers()
{
  triton::common::TritonJson::Value config_outputs;
  RETURN_IF_ERROR(
  model_state_->ModelConfig().MemberAsArray("output", &config_outputs));
  // std::vector<std::pair<std::string, std::int64_t>> outputs_bytes;
  for (size_t i = 0; i < config_outputs.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(config_outputs.IndexAsObject(i, &io));
      std::string io_name;
      io.MemberAsString("name", &io_name);
      std::string io_data_type;
      io.MemberAsString("data_type", &io_data_type);
      common::TritonJson::Value model_config_dims;
      common::TritonJson::Value reshape;
      if (io.Find("reshape", &reshape)) {
        reshape.MemberAsArray("shape", &model_config_dims);
        //todo: deal with reshape operations.
        //
        //
      } else {
        io.MemberAsArray("dims", &model_config_dims);
        std::vector<int64_t> dim_vec;
        int64_t byte_size;
        RETURN_IF_ERROR(DimsJsonToDimVec(model_config_dims, &dim_vec));
        std::vector<int64_t> dim_vec_with_mbs;
        // dim_vec_with_mbs.push_back(model_state_->MaxBatchSize());
        dim_vec_with_mbs.insert(
            dim_vec_with_mbs.end(), dim_vec.begin(), dim_vec.end());
        byte_size = GetByteSize(ModelConfigDataTypeToTritonServerDataType(io_data_type), dim_vec_with_mbs);
        outputs_bytes.push_back(std::make_pair(io_name,byte_size));
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,(
        std::string("\n io_name : ")+
        io_name+
        std::string("\n byte_size : ")+

        std::to_string(byte_size)).c_str());
        // std::cout<< "1111111111111111111111111111111 "<<std::endl;
        // std::cout<< "bytesize: "<< std::to_string(byte_size)<<std::endl<<std::flush;
        // std::cout<< "dim_vec_with_mbs :"<<std::endl;
        //     for(auto&dim :dim_vec_with_mbs){
        //       std::cout<< std::to_string(dim) <<", ";
        //     }
        // std::cout<< std::endl<<std::flush;
      }
    }
  int64_t max_byte_size = 0;
  for(auto& i :outputs_bytes){
    auto&j =i.second;
    max_byte_size += std::max((int64_t)1, j);
  }
  // std::cout <<std::string("=================================")<<std::endl<< max_byte_size <<std::endl<<std::flush;//================================= 3265920
  for(int i=0;i<model_state_->MaxBatchSize();i++){ // warning: maxbatchsize should be less then request_num in every request.
    IOBindingInfo io_binding_info;
    void* buffer = nullptr;
    buffer = malloc(std::max((int64_t)1,max_byte_size));

    // std::cout<< "1111111111111111111111111111111 "<<std::endl;
    // std::cout<< "malloc: "<< std::to_string(max_byte_size)<<std::endl<<std::flush;

    io_binding_info.byte_size_ = max_byte_size;
    io_binding_info.buffer_ = buffer;
    io_binding_info.device_buffer_ = buffer;
    io_binding_infos_.push_back(io_binding_info);
  }
  RETURN_IF_ERROR(InitializeConfigShapeOutputBindings(config_outputs));
  return nullptr;
}

//////////////////////////////////////////////////////
extern "C" {
// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // This backend does not require any "global" state but as an
  // example create a string to demonstrate.
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  // Delete the "global" state associated with the backend.
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

/////////////////////////////////////////

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));


  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}
/////////////////////////////////////////////////
// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());
  //todo : rk3588 has 3cores.so ...

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'rk' backend only supports NPU instances"));



  return nullptr;  // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

///////////////////////////////////////////////


// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Collect various timestamps during the execution of this batch or
  // requests. These values are reported below before returning from
  // the function.

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Best practice for a high-performance
  // implementation is to avoid introducing mutex/lock and instead use
  // only function-local and model-instance-specific state.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.
  
  //rk defaut set to support batching.
  // bool supports_batching = false;
  // RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));
  
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // 'responses' is initialized as a parallel array to 'requests',
  // with one TRITONBACKEND_Response object for each
  // TRITONBACKEND_Request object. If something goes wrong while
  // creating these response objects, the backend simply returns an
  // error from TRITONBACKEND_ModelInstanceExecute, indicating to
  // Triton that this backend did not create or send any responses and
  // so it is up to Triton to create and send an appropriate error
  // response for each request. RETURN_IF_ERROR is one of several
  // useful macros for error handling that can be found in
  // backend_common.h.

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // At this point, the backend takes ownership of 'requests', which
  // means that it is responsible for sending a response for every
  // request. From here, even if something goes wrong in processing,
  // the backend must return 'nullptr' from this function to indicate
  // success. Any errors and failures must be communicated via the
  // response objects.
  //
  // To simplify error handling, the backend utilities manage
  // 'responses' in a specific way and it is recommended that backends
  // follow this same pattern. When an error is detected in the
  // processing of a request, an appropriate error response is sent
  // and the corresponding TRITONBACKEND_Response object within
  // 'responses' is set to nullptr to indicate that the
  // request/response has already been handled and no futher processing
  // should be performed for that request. Even if all responses fail,
  // the backend still allows execution to flow to the end of the
  // function. RESPOND_AND_SET_NULL_IF_ERROR, and
  // RESPOND_ALL_AND_SET_NULL_IF_ERROR are macros from
  // backend_common.h that assist in this management of response
  // objects.

  // The backend could iterate over the 'requests' and process each
  // one separately. But for performance reasons it is usually
  // preferred to create batched input tensors that are processed
  // simultaneously. This is especially true for devices like GPUs
  // that are capable of exploiting the large amount parallelism
  // exposed by larger data sets.
  //
  // The backend utilities provide a "collector" to facilitate this
  // batching process. The 'collector's ProcessTensor function will
  // combine a tensor's value from each request in the batch into a
  // single contiguous buffer. The buffer can be provided by the
  // backend or 'collector' can create and manage it. In this backend,
  // there is not a specific buffer into which the batch should be
  // created, so use ProcessTensor arguments that cause collector to
  // manage it.

  BackendInputCollector collector(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      false /* pinned_enabled */, nullptr /* stream*/);

  // To instruct ProcessTensor to "gather" the entire batch of IN0
  // input tensors into a single contiguous buffer in CPU memory, set
  // the "allowed input types" to be the CPU ones (see tritonserver.h
  // in the triton-inference-server/core repo for allowed memory
  // types).
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types =
      {
        // {TRITONSERVER_MEMORY_CPU_PINNED, 0}, 
        {TRITONSERVER_MEMORY_CPU, 0}
      };

  const char* input_buffer;
  size_t input_buffer_byte_size;
  TRITONSERVER_MemoryType input_buffer_memory_type;
  int64_t input_buffer_memory_type_id;

  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      collector.ProcessTensor(
          model_state->InputTensorName().c_str(), nullptr /* existing_buffer */,
          0 /* existing_buffer_byte_size */, allowed_input_types, &input_buffer,
          &input_buffer_byte_size, &input_buffer_memory_type,
          &input_buffer_memory_type_id));

  // Finalize the collector. If 'true' is returned, 'input_buffer'
  // will not be valid until the backend synchronizes the CUDA
  // stream or event that was used when creating the collector. For
  // this backend, GPU is not supported and so no CUDA sync should
  // be needed; so if 'true' is returned simply log an error.
  const bool need_cuda_input_sync = collector.Finalize();
  if (need_cuda_input_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'rk' backend: does not suppory async required by collector");
  }

  // 'input_buffer' contains the batched "IN0" tensor. The backend can
  // implement whatever logic is necesary to produce "OUT0". This
  // backend simply returns the IN0 value in OUT0 so no actual
  // computation is needed.

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ": requests in batch " +
       std::to_string(request_count))
          .c_str());
  std::string tstr;
  IGNORE_ERROR(BufferAsTypedString(
      tstr, input_buffer, input_buffer_byte_size, model_state->TensorDataType()));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("batched " + model_state->InputTensorName() + " value: ignored")).c_str());
      //  tstr).c_str());

  //step 1. query rknn and verify input and output num.
  rknn_input_output_num io_num;
  rknn_context* rkctx=instance_state->getRknnContext();
  if(rkctx==NULL)RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses,request_count,TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "fail to rknn_query in out nums."));
  int ret=-1;
  ret=rknn_query(*rkctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if(ret<0){
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses,request_count,TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "fail to rknn_query in out nums."));
  }
  
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("model input num: ")+std::to_string(io_num.n_input)+
    std::string(", output num: ")+std::to_string(io_num.n_output)+std::string(" \n")).c_str());
  
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (uint i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret                  = rknn_query(*rkctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses,request_count,TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "fail to rknn_query in out attrs."));
    }
    dump_tensor_attr(&(input_attrs[i]));
    // index=0, name=images, n_dims=4, dims=[1, 384, 640, 3], n_elems=737280, size=737280, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  }
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (uint i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret                   = rknn_query(*rkctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
    // index=0, name=output, n_dims=4, dims=[1, 81, 48, 80], n_elems=311040, size=311040, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=55, scale=0.141896
    // index=1, name=376, n_dims=4, dims=[1, 81, 24, 40], n_elems=77760, size=77760, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=45, scale=0.142525
    // index=2, name=377, n_dims=4, dims=[1, 81, 12, 20], n_elems=19440, size=19440, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=53, scale=0.104938
  }

  //step 2 verify model argument is or not compatible.
  if(!verifyInputModelInput(input_attrs,input_buffer,request_count,input_buffer_byte_size)){
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses,request_count,TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "fail to verify model config and input tensors."));
  }
  

  //step 3 do infer. notice that here we use NHWC.
  //3.1 get rknn model arg.
  int channel = 3;
  int width   = 0;
  int height  = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,std::string("model is NCHW input fmt").c_str()); 

    channel = input_attrs[0].dims[1];
    width   = input_attrs[0].dims[2];
    height  = input_attrs[0].dims[3];
  } else {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,std::string("model is NHWC input fmt").c_str());
    width   = input_attrs[0].dims[1];
    height  = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
    // RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses,request_count,TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, std::string("tensor shape mismatch.").c_str()));
  }
  //model input height=640, width=384, channel=3
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("model input height="+std::to_string(height)+std::string(", width=")+
      std::to_string(width)+std::string(", channel=")+std::to_string(channel)).c_str()));

  //3.2 allocate input.
  //assume 1 input per request, so there are 1*request_nums inputs.
  rknn_input* inputs= new rknn_input[request_count];
  // memset(inputs, 0, sizeof(inputs));
  for(uint rc=0;rc<request_count;rc++){
    inputs[rc].index        = rc;
    // inputs[0].type       = RKNN_TENSOR_UINT8;
    inputs[rc].type         = getRKType(model_state->TensorDataType());
    inputs[rc].size         = width * height * channel;
    // inputs[rc].fmt       = RKNN_TENSOR_NHWC;
    inputs[rc].fmt          = input_attrs[0].fmt;
    inputs[rc].pass_through = 0;
    //3.2.1 assign input pointer
    inputs[rc].buf = (void*)(input_buffer+(rc*input_buffer_byte_size/request_count));
  }
  rknn_inputs_set(*rkctx, io_num.n_input, inputs);
  
  //3.3 allocate output 
  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (uint32_t i = 0; i < io_num.n_output && i<(uint32_t)model_state->MaxBatchSize() ; i++) {
    outputs[i].want_float = 0;
    outputs[i].is_prealloc = 1;
    outputs[i].index = i;
    outputs[i].buf = instance_state->io_binding_infos_[i].buffer_;
    outputs[i].size = instance_state->io_binding_infos_[i].byte_size_;
    memset(outputs[i].buf, 0, outputs[i].size);
  }

  //3.4 run  
  ret = rknn_run(*rkctx, NULL);
  if (ret < 0) {
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses,request_count,TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "fail to rknn_run."));
  }
  //3.5 get and copy output to response.
  //3.5.1 get output
  ret = rknn_outputs_get(*rkctx, io_num.n_output, outputs, NULL);
  if (ret < 0) {
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses,request_count,TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "fail to rknn_outputs_get."));
  }
  //3.5.2 delete inputs
  delete[] inputs;

  //3.5.3 copy to output_buffer
  const char* output_buffer = (const char* )instance_state->io_binding_infos_[0].buffer_;


  // const char*  output_buffer = nullptr;
  TRITONSERVER_MemoryType output_buffer_memory_type = input_buffer_memory_type;
  int64_t output_buffer_memory_type_id = input_buffer_memory_type_id;

  // // Only need an response tensor for requested outputs.
  // if ((response != nullptr) &&
  //     (request_required_outputs[idx].find(name) !=
  //       request_required_outputs[idx].end())) {
  //   TRITONBACKEND_Output* response_output = nullptr;
  //   RESPOND_AND_SET_NULL_IF_ERROR(
  //       &response, TRITONBACKEND_ResponseOutput(
  //                       response, &response_output, name.c_str(), dt,
  //                       batchn_shape.data(), batchn_shape.size()));
  //   cuda_copy |= SetOutputShapeTensorBuffer(
  //       shape_value_ptr, &response, response_output, tensor_element_cnt,
  //       batchn_shape[0], stream_);
  // }


  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  bool supports_first_dim_batching;
  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count,
      model_state->SupportsFirstDimBatching(&supports_first_dim_batching));
  std::vector<int64_t> tensor_shape;
  RESPOND_ALL_AND_SET_NULL_IF_ERROR(
      responses, request_count, model_state->TensorShape(tensor_shape));
  {
    std::ostringstream oss;
    oss <<"[";
    for (auto& i:tensor_shape){
      oss<<std::to_string(i);
      oss<<",";
    }
    oss.seekp(-1, std::ios_base::end);oss<<"]";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("now dump supports_first_dim_batching ")+std::to_string(supports_first_dim_batching)+
      std::string(" ,input_tensor_shape")+  oss.str()+std::string(",responses.size :[")+std::to_string(responses.size())+std::string("]")).c_str());
    
  }

  // for(auto& n:request_required_outputs){
  //   for(auto& m:n)
  //     LOG_MESSAGE(TRITONSERVER_LOG_ERROR,(std::string("'k': ")+std::string(m)).c_str());
  // }

  // Only need an response tensor for requested outputs.
  // for (size_t idx = 0; idx < request_count; idx++){
  //   const auto& request = requests[idx];
  //   auto& response = responses[idx];
  //   if ((response != nullptr) &&
  //     (request_required_outputs[idx].find(name) !=
  //       request_required_outputs[idx].end())) {
  //   TRITONBACKEND_Output* response_output = nullptr;
  //   RESPOND_AND_SET_NULL_IF_ERROR(
  //       &response, TRITONBACKEND_ResponseOutput(
  //                       response, &response_output, name.c_str(), dt,
  //                       batchn_shape.data(), batchn_shape.size()));
  //   cuda_copy |= SetOutputShapeTensorBuffer(
  //       shape_value_ptr, &response, response_output, tensor_element_cnt,
  //       batchn_shape[0], stream_);
  // }
  // }
  
  



  // Because the output tensor values are concatenated into a single
  // contiguous 'output_buffer', the backend must "scatter" them out
  // to the individual response output tensors.  The backend utilities
  // provide a "responder" to facilitate this scattering process.

  // The 'responders's ProcessTensor function will copy the portion of
  // 'output_buffer' corresonding to each request's output into the
  // response for that request.

  BackendOutputResponder responder(
      requests, request_count, &responses, model_state->TritonMemoryManager(),
      supports_first_dim_batching, false /* pinned_enabled */,
      nullptr /* stream*/);

  
  
  //3.5.4  make output response.
  // Collect the names of requested outputs. Do not include outputs
  // for requests that have already responded with an error.
  // std::vector<std::set<std::string>> request_required_outputs(request_count);
  for (size_t idx = 0; idx < request_count; idx++) {
    
    const auto& request = requests[idx];
    auto& response = responses[idx];
    auto& iobind_=instance_state->io_binding_infos_[idx];
    // LOG_MESSAGE(TRITONSERVER_LOG_INFO,(std::string("responder.ProcessTensor 11111111111111111111111111111111111-")+std::to_string(response==nullptr)).c_str()); 
    if (response != nullptr) {
      // LOG_MESSAGE(TRITONSERVER_LOG_INFO,std::string("responder.ProcessTensor 2222222222222222").c_str());
      uint32_t output_count;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_RequestOutputCount(request, &output_count));
      if (response != nullptr) {
        // LOG_MESSAGE(TRITONSERVER_LOG_INFO,std::string("responder.ProcessTensor 3333333333333333333333333").c_str());
        // for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
        for(auto&output_name:model_state->OutputTensorName()){
          // const char* output_name;
          // RESPOND_AND_SET_NULL_IF_ERROR(
          //     &response, TRITONBACKEND_RequestOutputName(
          //                    request, output_idx, &output_name));
          // LOG_MESSAGE(TRITONSERVER_LOG_INFO,std::string("output_shape_: [").c_str());
          // {
          //   std::stringstream ss;
          //   for (auto it = model_state->getOutputshapes(output_name).begin(); it != model_state->getOutputshapes(output_name).end(); it++)    {
          //       if (it != model_state->getOutputshapes(output_name).begin()) {
          //           ss << " ";
          //       }
          //       ss << *it;
          //   }
        
          //   std::cout << ss.str() << std::endl;
          // }
          // LOG_MESSAGE(TRITONSERVER_LOG_INFO,std::string("]").c_str());
          // LOG_MESSAGE(TRITONSERVER_LOG_INFO,std::string("responder.ProcessTensor 4444444444444444444444444444444").c_str());
          // if(model_state->OutputTensorName().find(output_name)!=model_state->OutputTensorName().end()){
          if(std::find(model_state->OutputTensorName().begin(), model_state->OutputTensorName().end(), output_name) != model_state->OutputTensorName().end()){
            // LOG_MESSAGE(TRITONSERVER_LOG_INFO,std::string("responder.ProcessTensor 555555555555555555555555555").c_str());
            TRITONBACKEND_Output* response_output = nullptr;
            TRITONSERVER_DataType dt = model_state->OutputTensorDataType(output_name);
            // To demonstrate response parameters we attach some here. Most
            // responses do not use parameters but they provide a way for
            // backends to communicate arbitrary information along with the
            // response.
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSetStringParameter(
                    response, "param0", "an example string parameter"),
                "failed setting string parameter");
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSetIntParameter(response, "param1", 42),
                "failed setting integer parameter");
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSetBoolParameter(response, "param2", false),
                "failed setting boolean parameter");
            size_t tensor_offset = 0;
            // const size_t tensor_byte_size = GetByteSize(dt, model_state->getOutputshapes(output_name));

            // TRITONBACKEND_Output* response_output;
            if (response != nullptr) {
              uint32_t output_count;
              RESPOND_AND_SET_NULL_IF_ERROR(
                  &response, TRITONBACKEND_RequestOutputCount(request, &output_count));
              for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
                // const char* name;
                // RESPOND_AND_SET_NULL_IF_ERROR(
                //     &response,
                //     TRITONBACKEND_RequestOutputName(request, output_idx, &name));
                // if ((response != nullptr) && (output_name == name)) {
                  RESPOND_AND_SET_NULL_IF_ERROR(
                      &response, TRITONBACKEND_ResponseOutput(
                                    response, &response_output, output_name.c_str(), dt,
                                    model_state->getOutputshapes(output_name).data(), model_state->getOutputshapes(output_name).size()));
                  //创建output_buffer
                  void* output_buffer;
                  TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
                  int64_t output_memory_type_id = 0;
                  RESPOND_AND_SET_NULL_IF_ERROR(
                      &response, TRITONBACKEND_OutputBuffer(
                          response_output, &output_buffer, GetByteSize(dt, model_state->getOutputshapes(output_name)), &output_memory_type,
                          &output_memory_type_id));
                  memcpy(output_buffer, (void*)(((char*)iobind_.buffer_+tensor_offset)), GetByteSize(dt, model_state->getOutputshapes(output_name))); 
                  // if (response != nullptr) {
                  //   responder.SetFixedSizeBuffer(
                  //       &response, response_output, output_name, tensor_byte_size,
                  //       tensor_offset, buffer, memory_type, memory_type_id,
                  //       use_pinned_memory_type, false /* state */);
                  // }

                  break;
                // }
                tensor_offset += GetByteSize(dt, model_state->getOutputshapes(output_name));
              }
            }

            
            
          }
        }
      }
    }
  }
  

  // Finalize the responder. If 'true' is returned, the OUT0
  // tensors' data will not be valid until the backend synchronizes
  // the CUDA stream or event that was used when creating the
  // responder. For this backend, GPU is not supported and so no
  // CUDA sync should be needed; so if 'true' is returned simply log
  // an error.

  const bool need_cuda_output_sync = responder.Finalize();
  if (need_cuda_output_sync) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        "'minimal' backend: unexpected CUDA sync required by responder");
  }

  

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send response");
    }
  }
  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
  // For batch statistics need to know the total batch size of the
  // requests. This is not necessarily just the number of requests,
  // because if the model supports batching then any request can be a
  // batched request itself.
  size_t total_batch_size = 0;
  if (!supports_first_dim_batching) {
    total_batch_size = request_count;
  } else {
    for (uint32_t r = 0; r < request_count; ++r) {
      auto& request = requests[r];
      TRITONBACKEND_Input* input = nullptr;
      LOG_IF_ERROR(
          TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
          "failed getting request input");
      if (input != nullptr) {
        const int64_t* shape = nullptr;
        LOG_IF_ERROR(
            TRITONBACKEND_InputProperties(
                input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr),
            "failed getting input properties");
        if (shape != nullptr) {
          total_batch_size += shape[0];
        }
      }
    }
  }
#else
  (void)exec_start_ns;
  (void)exec_end_ns;
  (void)compute_start_ns;
  (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

  // Done with the request objects so release them.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    // Before releasing, record failed requests as those where
    // responses[r] is nullptr. The timestamps are ignored in this
    // case.
    if (responses[r] == nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              instance_state->TritonModelInstance(), request,
              false /* success */, 0, 0, 0, 0),
          "failed reporting request statistics");
    }

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;  // success
}

}  // extern "C"

}}}