import chainer
import chainer.functions as F
import chainer.links as L


class CAE(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal()
        super(CAE, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 3, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(64, 128, 4, 2, 1, initialW=w)
            self.conv3 = L.Convolution2D(128, 256, 4, 2, 1, initialW=w)

            self.deconv3 = L.Deconvolution2D(256, 128, 4, 2, 1, initialW=w)
            self.deconv2 = L.Deconvolution2D(128, 64, 4, 2, 1, initialW=w)
            self.deconv1 = L.Deconvolution2D(64, 3, 3, 1, 1, initialW=w)

    def encode(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h)
        h = self.conv2(h)
        h = F.leaky_relu(h)
        h = self.conv3(h)
        h = F.leaky_relu(h)

        return h

    def decode(self, x):
        h = self.deconv3(x)
        h = F.relu(h)
        h = self.deconv2(h)
        h = F.relu(h)
        h = self.deconv1(h)

        return h

    def __call__(self, x):
        h = self.encode(x)
        h = self.decode(h)
        h = F.tanh(h)

        return h


class DeepCAE(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal()
        super(DeepCAE, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(3, 64, 3, 1, 1, initialW=w)
            self.conv1 = L.Convolution2D(64, 128, 4, 2, 1, initialW=w)
            self.conv2 = L.Convolution2D(128, 256, 4, 2, 1, initialW=w)
            self.conv3 = L.Convolution2D(256, 512, 4, 2, 1, initialW=w)
            self.conv4 = L.Convolution2D(512, 512, 4, 2, 1, initialW=w)
            self.conv5 = L.Convolution2D(512, 512, 4, 2, 1, initialW=w)
            # self.conv6 = L.Convolution2D(512, 512, 4, 2, 1, initialW=w)
            # self.conv7 = L.Convolution2D(512, 512, 4, 2, 1, initialW=w)

            # self.deconv7 = L.Convolution2D(512, 512, 3, 1, 1, initialW=w)
            # self.deconv6 = L.Convolution2D(512, 512, 3, 1, 1, initialW=w)
            self.deconv5 = L.Convolution2D(512, 512, 3, 1, 1, initialW=w)
            self.deconv4 = L.Convolution2D(512, 512, 3, 1, 1, initialW=w)
            self.deconv3 = L.Convolution2D(512, 256, 3, 1, 1, initialW=w)
            self.deconv2 = L.Convolution2D(256, 128, 3, 1, 1, initialW=w)
            self.deconv1 = L.Convolution2D(128, 64, 3, 1, 1, initialW=w)
            self.deconv0 = L.Convolution2D(64, 3, 3, 1, 1, initialW=w)

    def encode(self, x):
        h = self.conv0(x)
        h = F.leaky_relu(h)
        h = self.conv1(h)
        h = F.leaky_relu(h)
        h = self.conv2(h)
        h = F.leaky_relu(h)
        h = self.conv3(h)
        h = F.leaky_relu(h)
        h = self.conv4(h)
        h = F.leaky_relu(h)
        h = self.conv5(h)
        h = F.leaky_relu(h)
        # h = self.conv6(h)
        # h = F.leaky_relu(h)
        # h = self.conv7(h)
        # h = F.leaky_relu(h)

        return h

    def decode(self, x):
        h = F.unpooling_2d(x, 2, 2, cover_all=False)
        # h = self.deconv7(h)
        # h = F.relu(h)
        # h = F.unpooling_2d(h, 2, 2, cover_all=False)
        # h = self.deconv6(h)
        # h = F.relu(h)
        # h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = self.deconv5(h)
        h = F.relu(h)
        h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = self.deconv4(h)
        h = F.relu(h)
        h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = self.deconv3(h)
        h = F.relu(h)
        h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = self.deconv2(h)
        h = F.relu(h)
        h = F.unpooling_2d(h, 2, 2, cover_all=False)
        h = self.deconv1(h)
        h = F.relu(h)
        h = self.deconv0(h)

        return h

    def __call__(self, x):
        h = self.encode(x)
        h = self.decode(h)
        h = F.tanh(h)

        return h
