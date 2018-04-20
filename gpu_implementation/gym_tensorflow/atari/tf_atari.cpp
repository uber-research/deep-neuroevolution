#include <iostream>
#include <string>
#include <ale_interface.hpp>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "../tf_env.h"

#ifdef __USE_SDL
  #include <SDL.h>
#endif

using namespace tensorflow;
using namespace std;
using namespace ale;

#define RAM_SIZE (128)

class AtariEnvironment : public Environment<uint8>, public StepInterface<int>
{
    public:
        AtariEnvironment(int batch_size)
        {
            m_numNoops.resize(batch_size, 0);
            m_maxFrames.resize(batch_size, 100000);
            m_pInterfaces = new ALEInterface[batch_size];
        }
        void load_rom(string game, int i)
        {
            assert(m_numNoops[i] == 0);
            m_numNoops[i] = 1;
            m_pInterfaces[i].setFloat("repeat_action_probability", 0.0f);
            m_pInterfaces[i].setInt("random_seed", 0);
            m_pInterfaces[i].loadROM(game);
        }
        virtual ~AtariEnvironment() {
            delete[] m_pInterfaces;
        }

        TensorShape get_action_shape() override
        {
            return TensorShape();
        }

        TensorShape get_observation_shape() override
        {
            return TensorShape({2,
                                static_cast<int>(m_pInterfaces[0].getScreen().height()),
                                static_cast<int>(m_pInterfaces[0].getScreen().width())});
        }

        void get_observation(uint8 *data, int idx) override
        {
            const auto ssize = m_pInterfaces[idx].getScreen().height() * m_pInterfaces[idx].getScreen().width();
            memcpy(data, m_pInterfaces[idx].theOSystem->console().mediaSource().previousFrameBuffer(), ssize);
            memcpy(data + ssize, m_pInterfaces[idx].theOSystem->console().mediaSource().currentFrameBuffer(), ssize);
        }

        float step(int idx, const int* action) override
        {
            int rewards = 0;
            for (int i = 0; i < m_repeat; ++i)
            {
                assert(m_pInterfaces[idx].getMinimalActionSet().size() > (*action));
                rewards += m_pInterfaces[idx].act(m_pInterfaces[idx].getMinimalActionSet()[*action]);
                if (is_done(idx))
                    break;
            }
            return rewards;
        }

        bool is_done(int idx) override
        {
            return m_pInterfaces[idx].game_over() ||
                   m_pInterfaces[idx].getEpisodeFrameNumber() - m_numNoops[idx] >= m_maxFrames[idx];
        }

        void reset(int i, int numNoops=0, int maxFrames=100000) override
        {
            m_pInterfaces[i].reset_game();
            if(numNoops > 0)
            {
                assert(m_pInterfaces[i].getMinimalActionSet()[0] == Action::PLAYER_A_NOOP);
                for (int s = 0; s < numNoops;++s)
                {
                    m_pInterfaces[i].act(Action::PLAYER_A_NOOP);
                    if (m_pInterfaces[i].game_over())
                        m_pInterfaces[i].reset_game();
                }
            }
            // Check if FIRE is part of the minimal action set
            if (m_pInterfaces[i].getMinimalActionSet()[1] == Action::PLAYER_A_FIRE)
            {
                assert(m_pInterfaces[i].getMinimalActionSet().size() >= 3);
                int action = 1;
                step(i, &action);
                if (m_pInterfaces[i].game_over())
                    m_pInterfaces[i].reset_game();

                action = 2;
                step(i, &action);
                if (m_pInterfaces[i].game_over())
                    m_pInterfaces[i].reset_game();
            }
            m_numNoops[i] = m_pInterfaces[i].getEpisodeFrameNumber();
            m_maxFrames[i] = maxFrames;
        }

        void get_final_state(float *data, int idx)
        {
            auto ram = m_pInterfaces[idx].getRAM();
            for (auto i = 0; i < RAM_SIZE; ++ i)
                data[i] = ram.get(i);
        }

        string DebugString() override { return "AtariEnvironment"; }
      private:
        ALEInterface* m_pInterfaces;
        bool m_initialized;
        int m_repeat = 4;
        std::vector<int> m_numNoops;
        std::vector<int> m_maxFrames;
};

class AtariMakeOp : public EnvironmentMakeOp {
    public:
    explicit AtariMakeOp(OpKernelConstruction* context) : EnvironmentMakeOp(context) {
        OP_REQUIRES_OK(context, context->GetAttr("game", &m_game));
        ale::Logger::setMode(ale::Logger::mode(2));
    }

 private:
    virtual Status CreateResource(OpKernelContext* context, BaseEnvironment** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        AtariEnvironment* env = new AtariEnvironment(batch_size);
        if (env == nullptr)
            return errors::ResourceExhausted("Failed to allocate");
        *ret = env;

        const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
        const int num_threads = std::min(thread_pool->num_threads, batch_size);
        auto f = [&](int thread_id) {
            for(int b =thread_id; b < batch_size;b+=num_threads)
            {
                env->load_rom(m_game, b);
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
    std::string m_game;
};

REGISTER_OP("AtariMake")
    .Attr("batch_size: int")
    .Attr("game: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("handle: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("AtariMake").Device(DEVICE_CPU), AtariMakeOp);
