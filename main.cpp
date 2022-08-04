#include <cstdlib>
#include <iostream>
#include <cmath>
#include <robot_dart/control/simple_control.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#include <torch/torch.h>
#include <memory>
#include <iomanip>
#include <math.h>       /* sin */
#include <random>
#ifdef GRAPHIC
#include <robot_dart/gui/magnum/graphics.hpp>
#endif
auto cuda_available = torch::cuda::is_available();
torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
std::default_random_engine generator;

namespace rd = robot_dart;

class pendulum {

public:
    torch::Tensor _state;
    //bool done;
    //double reward;
    int simusteps=0;
    std::shared_ptr<robot_dart::Robot> robot = std::make_shared<robot_dart::Robot>("pendulum.urdf");
    std::shared_ptr<robot_dart::RobotDARTSimu> simu = std::make_shared<robot_dart::RobotDARTSimu>(0.001);

    pendulum()
    {
        robot->fix_to_world();
        robot->set_position_enforced(true);
        robot->set_positions(robot_dart::make_vector({M_PI}));
        robot->set_actuator_types("torque");
        simu->add_robot(robot);
#ifdef GRAPHIC
        auto graphics = std::make_shared<robot_dart::gui::magnum::Graphics>();
            simu->set_graphics(graphics);
#endif
        //_state = torch::tensor({M_PI}, torch::kDouble);
    }

    std::tuple<torch::Tensor, double, bool> step(torch::Tensor act)
    {
        bool success;
        auto act_a = act.accessor<double, 1>();
        double move = act_a[0];

        auto cmds = rd::make_vector({move});

        for (int i = 0; i < 50; i++) {
            robot->set_commands(cmds);
            simu->step_world();
            simusteps++;
        }

        double temp_pos = robot->positions()[0];
        //double data[] = {temp_pos};
        double sin_pos=sin(robot->positions()[0]);
        double cos_pos=cos(robot->positions()[0]);
        bool done = false;
        double reward;
        double temp_velocity = robot->velocities()[0];
        double theta = angle_dist(temp_pos, 0.);
        // reward=-(std::abs(M_PI-robot->positions()[0]));
        reward = -theta;
        //reward = -std::abs(temp_velocity);

        //if (std::abs(M_PI-temp_pos)<0.0001) {
        if ((theta)<0.1){
            //if ((std::abs(theta)<0.1)){

            //auto cmds = rd::make_vector({0});
            //robot->set_commands(cmds);
            done = false;
            reward = 10;
            //simusteps = 0;
            //std::cout << "success"<<std::endl;
            success=true;
            //std::cout<<temp_pos<<std::endl;

            //torch::Tensor reset();
        }
        if (simusteps == 5000) {
            //auto cmds = rd::make_vector({0});
            //robot->set_commands(cmds);
            done = true;
            simusteps = 0;
            std::cout << "fail"<<std::endl;
            success=false;
            std::cout<<theta<<std::endl;
            //torch::Tensor reset();
        }

        //_state = torch::from_blob(data, {1}, torch::TensorOptions().dtype(torch::kDouble));
        _state = torch::tensor({sin_pos, cos_pos, temp_velocity}, torch::kDouble);

        auto _stateNan = at::isnan(_state).any().item<bool>();

        if (_stateNan==1){
            std::cout<<"_statesNan"<<_state<<std::endl;
            std::cout<<"mu nan"<<std::endl;

            exit (EXIT_FAILURE);
        }


        return {_state, reward, done};
    }

    torch::Tensor reset()
    {
        simusteps = 0;
        robot->reset();
        robot->set_positions(robot_dart::make_vector({M_PI}));
        //double tempor =robot->positions()[0];
        _state = torch::tensor({0.0,-1.0,0.0}, torch::kDouble);
        return _state;
    }

    static double angle_dist(double a, double b)
    {
        double theta = b - a;
        while (theta < -M_PI)
            theta += 2 * M_PI;
        while (theta > M_PI)
            theta -= 2 * M_PI;
        return std::abs(theta);
    }
};

class critic : public torch::nn::Module {
public:
    torch::nn::Linear fc1, out;

    critic(int inputs, int hidden_size, int n_a)
            : fc1(register_module("fc1", torch::nn::Linear(inputs, hidden_size))),
            //fc2(register_module("fc2", torch::nn::Linear(hidden_size, n_a))),
              out(register_module("out", torch::nn::Linear(hidden_size, 1))) {}

    torch::Tensor forward(torch::Tensor state)
    {
        //torch::Tensor x = torch::cat({state, action.unsqueeze(1)}, 1); //check for errors
        torch::Tensor x = torch::relu(fc1(state));
        x=out(x);

        return {x};
    }
};

class actor : public torch::nn::Module {
public:
    torch::nn::Linear fc1, out, out_logsigma;

    actor(int inputs, int hidden_size, int n_a)
            : fc1(register_module("fc1", torch::nn::Linear(inputs, hidden_size))),
              out(register_module("out", torch::nn::Linear(hidden_size, n_a))),
              out_logsigma(register_module("out_sigma", torch::nn::Linear(hidden_size, n_a))) {}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = torch::relu(fc1(x));
        torch::Tensor mu = out(x);
        torch::Tensor logsigma = out_logsigma(x);

        return {mu, logsigma};
    }
};

class Agent {
public:
    double gamma = 0.99;
    int n_outputs = 1;
    int n_actions = 1;
    int input_dims = 3;
    int layer1_size = 64;
    int layer2_size = 64;
    torch::Tensor log_probs;
    torch::optim::SGD *actor_optimizer;
    torch::optim::SGD *critic_optimizer;
    std::shared_ptr<actor> ac = std::make_shared<actor>(input_dims, layer1_size, n_actions);
    std::shared_ptr<critic> cr = std::make_shared<critic>(input_dims, layer1_size, n_actions);


    Agent(float alpha, float beta) {
        // Optimizer
        ac->to(torch::kDouble);
        cr->to(torch::kDouble);
        this->actor_optimizer = new torch::optim::SGD(ac->parameters(), alpha);
        this->critic_optimizer = new torch::optim::SGD(cr->parameters(), beta);
    }

    std::tuple<torch::Tensor, torch::Tensor> choose_action(torch::Tensor observation) {


        torch::Tensor logsigma;
        torch::Tensor mu;
        std::tie(mu, logsigma) = ac->forward(observation);
        observation=observation.detach();
        torch::Tensor sigma = torch::exp(logsigma);

        auto sample = torch::randn({1}, torch::kDouble) * sigma + mu;

        auto pdf = (1.0 / (sigma * std::sqrt(2 * M_PI))) * torch::exp(-0.5 * torch::pow((sample.detach() - mu) / sigma, 2));
        this->log_probs = torch::log(pdf);
        torch::Tensor action = torch::tanh(sample);

        return {action * 5, log_probs};
    };
};

int main() {
    std::cout << "FeedForward Neural Network\n\n";

    // Hyper parameters
    const int64_t input_size = 1;
    const int64_t hidden_size = 256;
    const int64_t num_classes = 2;
    const int64_t batch_size = 100;
    const size_t num_epochs = 5;
    const double alpha = 0.000005;
    const double beta = 0.00001;

    pendulum env;

    Agent *agent = new Agent(0.0001, 0.001);

    int num_episodes = 800000;
    bool done;
    double score;
    double reward;

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (int i = 0; i < num_episodes; i++){
        done = false;
        score = 0;
        torch::Tensor observation = env.reset();
        double I=1;

        while (!done){
            torch::Tensor action;
            torch::Tensor log_probs;

            std::tie(action, log_probs)  = agent->choose_action(observation);
            torch::Tensor observation_;
            std::tie(observation_, reward, done) = env.step(action);
            score += reward;
            torch::Tensor currentstateval = agent->cr->forward(observation);
            torch::Tensor newstateval = agent->cr->forward(observation_);

            torch::Tensor val_loss = torch::nn::functional::mse_loss(reward + agent->gamma * newstateval, currentstateval);
            val_loss *= I;
            auto advantage = reward + agent->gamma * newstateval - currentstateval;
            auto actor_loss = -log_probs * advantage;
            actor_loss =actor_loss * I;
            agent->actor_optimizer->zero_grad();
            actor_loss.backward({},c10::optional<bool>(true));
            agent->actor_optimizer->step();

            agent->critic_optimizer->zero_grad();
            val_loss.backward();
            agent->critic_optimizer->step();

            observation = observation_;
            I = I * agent->gamma;
        }

        printf("Episode %d score %.2f\n", i, score);
    }

    return 0;
};