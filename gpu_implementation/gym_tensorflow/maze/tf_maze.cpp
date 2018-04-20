/*
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tf_env.h"
#include "maze.h"

#ifdef __USE_SDL
  #include <SDL.h>
#endif

using namespace tensorflow;
using namespace std;

class MazeEnvironment : public Environment<float>, public StepInterface<float>
{
    public:
        MazeEnvironment(int batch_size)
        {
            m_pInterfaces = new maze::Environment[batch_size];
            m_numSteps.resize(batch_size, 0);
        }
        void load(std::string filename, int i)
        {
            m_pInterfaces[i].load_from(filename.c_str());
        }
        virtual ~MazeEnvironment() {
            delete[] m_pInterfaces;
        }

        TensorShape get_observation_shape() override
        {
            return TensorShape({m_pInterfaces[0].get_sensor_size()});
        }

        void get_observation(float *data, int idx) override
        {
            assert(idx < m_numSteps.size());
            m_pInterfaces[idx].generate_neural_inputs(data);
        }

        void get_final_state(float *data, int idx)
        {
            assert(idx < m_numSteps.size());
            data[0] = m_pInterfaces[idx].hero.location.x;
            data[1] = m_pInterfaces[idx].hero.location.y;
        }

        TensorShape get_action_shape() override {
            return TensorShape({2});
        }

        float step(int idx, const float* action) override {
            assert(idx < m_numSteps.size());
            m_pInterfaces[idx].interpret_outputs(float(action[0]) + 0.5, 0.5 + float(action[1]));
            m_pInterfaces[idx].Update();
            m_numSteps[idx] += 1;
            if (is_done(idx))
            {
                return -m_pInterfaces[idx].distance_to_target();
            }
            return 0.0f;
        }

        bool is_done(int idx) override {
            assert(idx < m_numSteps.size());
            return m_numSteps[idx] >= 400;
        }

        void reset(int i, int numNoops=0, int maxFrames=100000) override {
            m_pInterfaces[i].reset();
            m_numSteps[i] = 0;
        }

        string DebugString() override { return "MazeEnvironment"; }
      public:
        maze::Environment* m_pInterfaces;
        std::vector<int> m_numSteps;
};

class MazeMakeOp : public EnvironmentMakeOp {
    public:
    explicit MazeMakeOp(OpKernelConstruction* context) : EnvironmentMakeOp(context) {
        OP_REQUIRES_OK(context, context->GetAttr("filename", &m_filename));
    }

 private:
    virtual Status CreateResource(OpKernelContext* context, BaseEnvironment** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        MazeEnvironment* env = new MazeEnvironment(batch_size);
        if (env == nullptr)
            return errors::ResourceExhausted("Failed to allocate");
        *ret = env;

        const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
        const int num_threads = std::min(thread_pool->num_threads, batch_size);
        auto f = [&](int thread_id) {
            for(int b =thread_id; b < batch_size;b+=num_threads)
            {
                env->load(m_filename, b);
            }
        };

        BlockingCounter counter(num_threads-1);
        for (int i = 1; i < num_threads; ++i) {
            thread_pool->workers->Schedule([&, i]() {
                f(i);
                counter.DecrementCount();
            });
        }
        f(0);
        counter.Wait();
        return Status::OK();
    }
    std::string m_filename;
};

REGISTER_OP("MazeMake")
    .Attr("batch_size: int")
    .Attr("filename: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("handle: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("MazeMake").Device(DEVICE_CPU), MazeMakeOp);


class MazeFinalStateOp : public OpKernel {
    public:
    explicit MazeFinalStateOp(OpKernelConstruction* context) : OpKernel(context) {
    }
    void Compute(OpKernelContext* context) override {
        const Tensor &indices_tensor = context->input(1);
        auto indices_flat = indices_tensor.flat<int>();

        const int m_numInterfaces = indices_tensor.NumElements();

        Tensor* position_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                                            TensorShape({static_cast<int>(m_numInterfaces), 2}),
                                            &position_tensor));
        auto position_flat = position_tensor->flat<float>();
        if(m_numInterfaces > 0)
        {
            BaseEnvironment *tmp_env;
            OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0), &tmp_env));

            auto env = dynamic_cast<MazeEnvironment *>(tmp_env);
            OP_REQUIRES(context, env != nullptr, errors::Internal("BaseEnvironment is not of MazeEnvironment type."));
            core::ScopedUnref s(tmp_env);

            const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
            const int num_threads = std::min(thread_pool->num_threads, int(m_numInterfaces));

            auto f = [&](int thread_id) {
                // Set all but the first element of the output tensor to 0.
                for(int b =thread_id; b < m_numInterfaces;b+=num_threads)
                {
                    env->get_final_state(&position_flat(b * 2), indices_flat(b));
                }
            };

            BlockingCounter counter(num_threads-1);
            for (int i = 1; i < num_threads; ++i) {
                thread_pool->workers->Schedule([&, i]() {
                    f(i);
                    counter.DecrementCount();
                });
            }
            f(0);
            counter.Wait();
        }
    }
};

REGISTER_OP("MazeFinalState")
    .Input("handle: resource")
    .Input("indices: int32")
    .Output("position: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input));
        ::tensorflow::shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->Concatenate(input, c->MakeShape({2}), &output));
        c->set_output(0, output);
        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("MazeFinalState").Device(DEVICE_CPU), MazeFinalStateOp);

