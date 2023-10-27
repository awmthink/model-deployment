#include <torch/torch.h>

#include <iostream>

using torch::nn::Module;

struct DCGANGeneratorImpl : Module {
  DCGANGeneratorImpl(int noise_size)
      : conv1(
            torch::nn::ConvTranspose2dOptions(noise_size, 256, 4).bias(false)),
        bn1(256),
        conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        bn2(128),
        conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        bn3(64),
        conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    register_module("conv4", conv4);
    register_module("bn1", bn1);
    register_module("bn2", bn2);
    register_module("bn3", bn3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(bn1(conv1(x)));
    x = torch::relu(bn2(conv2(x)));
    x = torch::relu(bn3(conv3(x)));
    x = torch::tanh(conv4(x));
    return x;
  }

  torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
  torch::nn::BatchNorm2d bn1, bn2, bn3;
};
TORCH_MODULE(DCGANGenerator);

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    b = register_parameter("b", torch::randn({M}));
  }
  torch::Tensor forward(torch::Tensor input) { return linear(input) + b; }
  torch::Tensor b;
  torch::nn::Linear linear;
};

int main() {
  torch::Device device(torch::kCUDA, 0);
  const int64_t kBatchSize = 32;

  DCGANGenerator generator(100);
  torch::nn::Sequential discriminator(
      // Layer 1
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
      torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 2
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
      torch::nn::BatchNorm2d(128),
      torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 3
      torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4)
                            .stride(2)
                            .padding(1)
                            .bias(false)),
      torch::nn::BatchNorm2d(256),
      torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 4
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
      torch::nn::Sigmoid());

  generator->to(device);
  discriminator->to(device);

  auto dataset = torch::data::datasets::MNIST(
                     "/home/yangyansheng/workspace/pyml/data/MNIST/raw/")
                     .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                     .map(torch::data::transforms::Stack<>());

  const int64_t batches_per_epoch =
      std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(4));

  auto adam_options =
      torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5));

  torch::optim::Adam generator_optimizer(generator->parameters(), adam_options);

  torch::optim::Adam discriminator_optimizer(discriminator->parameters(),
                                             adam_options);

  int64_t kNumberOfEpochs = 5;

  for (int64_t epoch = 1; epoch < kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels =
          torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real =
          torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images.
      torch::Tensor noise =
          torch::randn({batch.data.size(0), 100, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake =
          torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss =
          torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();

      std::printf("\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f", epoch,
                  kNumberOfEpochs, ++batch_index, batches_per_epoch,
                  d_loss.item<float>(), g_loss.item<float>());
    }
  }

  return 0;
}