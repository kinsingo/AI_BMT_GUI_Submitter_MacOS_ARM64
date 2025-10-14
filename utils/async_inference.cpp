#include "async_inference.hpp"
#include "utils.hpp"

#if defined(__unix__)
#include <sys/mman.h>
#endif


static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size, void* buff = nullptr) {
    #if defined(__unix__)
        auto addr = mmap(buff, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (MAP_FAILED == addr) throw std::bad_alloc();
        return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
    #elif defined(_MSC_VER)
        auto addr = VirtualAlloc(buff, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!addr) throw std::bad_alloc();
        return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
    #else
    #pragma error("Aligned alloc not supported")
    #endif
}
size_t align_to_page_size(size_t size) {
    const size_t page_size = sysconf(_SC_PAGE_SIZE);  // For Unix-like systems
    return (size + page_size - 1) & ~(page_size - 1);  // Round up to the nearest page boundary
}

AsyncModelInfer::AsyncModelInfer(const std::string &hef_path,
                                 std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> results_queue)
{
    auto vdevice_exp = hailort::VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create VDevice, status = " << vdevice_exp.status() << std::endl;
        throw std::runtime_error("Failed to create VDevice");
    }
    this->vdevice = std::move(vdevice_exp.value()); 

    auto infer_model_exp = this->vdevice->create_infer_model(hef_path);
    if (!infer_model_exp) {
        std::cerr << "Failed to create infer model, status = " << infer_model_exp.status() << std::endl;
        throw std::runtime_error("Failed to create infer model");
    }
    

    this->infer_model = infer_model_exp.release();
    auto outputs = this->infer_model->outputs();
    for (auto& output : outputs) {
        output.set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    }
    this->input_buffer_guards.reserve(this->infer_model->inputs().size());
    this->output_buffer_guards.reserve(this->infer_model->outputs().size());

    for (auto& output_vstream_info : this->infer_model->hef().get_output_vstream_infos().release()) {
        std::string name(output_vstream_info.name);
        this->output_vstream_info_by_name[name] = output_vstream_info;
    }

    configure(results_queue);
}
void AsyncModelInfer::crt(){
    auto vdevice_exp = hailort::VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create VDevice, status = " << vdevice_exp.status() << std::endl;
        throw std::runtime_error("Failed to create VDevice");
    }
    this->vdevice = std::move(vdevice_exp.value()); 

}
void AsyncModelInfer::PathAndResult(const std::string &hef_path)
{
    
    auto infer_model_exp = this->vdevice->create_infer_model(hef_path);
    if (!infer_model_exp) {
        std::cerr << "Failed to create infer model, status = " << infer_model_exp.status() << std::endl;
        throw std::runtime_error("Failed to create infer model");
    }
    

    this->infer_model = infer_model_exp.release();
    auto outputs = this->infer_model->outputs();
    for (auto& output : outputs) {
        output.set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    }
    infer_model->set_batch_size(32);
    this->input_buffer_guards.reserve(this->infer_model->inputs().size());
    this->output_buffer_guards.reserve(this->infer_model->outputs().size());

    for (auto& output_vstream_info : this->infer_model->hef().get_output_vstream_infos().release()) {
        std::string name(output_vstream_info.name);
        this->output_vstream_info_by_name[name] = output_vstream_info;
    }

    //configure(results_queue);
}
const std::vector<hailort::InferModel::InferStream>& AsyncModelInfer::get_inputs(){
    return std::move(this->infer_model->inputs());
}

const std::vector<hailort::InferModel::InferStream>& AsyncModelInfer::get_outputs(){
    return std::move(this->infer_model->outputs());
}

const std::shared_ptr<hailort::InferModel> AsyncModelInfer::get_infer_model(){
    return this->infer_model;
}

void AsyncModelInfer::configure(std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> output_data_queue) { 

    this->configured_infer_model = this->infer_model->configure().expect("Failed to create configured infer model");
    this->bindings = configured_infer_model.create_bindings().expect("Failed to create infer bindings");
    this->output_data_queue = std::move(output_data_queue);
}

std::shared_ptr<BoundedTSQueue<InferenceOutputItem>> AsyncModelInfer::get_queue(){
    return output_data_queue;
}

void AsyncModelInfer::infer(std::shared_ptr<std::vector<uint8_t>> input_data, size_t frame_idx) 
{
    set_input_buffers(input_data);
    auto output_data_and_infos = prepare_output_buffers();
    wait_and_run_async(frame_idx, output_data_and_infos);
}

void AsyncModelInfer::set_input_buffers(const std::shared_ptr<std::vector<uint8_t>> &input_data)
{
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t frame_size = infer_model->input(input_name)->get_frame_size();
        //std::cout<<frame_size<<std::endl;

        auto status = bindings.input(input_name)->set_buffer(MemoryView(input_data->data(), frame_size));
        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed to set infer input buffer, status = " << status << std::endl;
        }
        input_buffer_guards.push_back(input_data);
    }
}

std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> AsyncModelInfer::prepare_output_buffers()
{
    std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> result;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t frame_size = infer_model->output(output_name)->get_frame_size();
        //size_t aligned_frame_size = align_to_page_size(frame_size);
        output_data_holder = page_aligned_alloc(frame_size);
        //std::cout <<frame_size<<std::endl;
        auto status = bindings.output(output_name)->set_buffer(MemoryView(output_data_holder.get(), frame_size));

        if (HAILO_SUCCESS != status) {
            std::cerr << "Failed to set infer output buffer, status = " << status << std::endl;
        }

        result.push_back(std::make_pair(
            bindings.output(output_name)->get_buffer()->data(),
            output_vstream_info_by_name[output_name]
        ));

        output_buffer_guards.push_back(output_data_holder);
    }

    return result;
}

void AsyncModelInfer::clear()
{
    input_buffer_guards.clear();
    output_buffer_guards.clear();
    output_data_holder.reset(); // page_aligned_alloc 한 output_data도 초기화
}

void AsyncModelInfer::wait_and_run_async(size_t frame_idx,
    const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &output_data_and_infos)
{
    auto status = configured_infer_model.wait_for_async_ready(std::chrono::milliseconds(1000));
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed wait_for_async_ready, status = " << status << std::endl;
    }
    InferenceOutputItem item;
    item.frame_idx = frame_idx;
    item.output_data_and_infos = output_data_and_infos;

    auto job = configured_infer_model.run_async(
        bindings,
        [this, item](const hailort::AsyncInferCompletionInfo& info)
        {
            get_queue()->push(item);
        }
    );

    if (!job) {
        std::cerr << "Failed to start async infer job, status = " << job.status() << std::endl;
    }

    job->detach();
}
