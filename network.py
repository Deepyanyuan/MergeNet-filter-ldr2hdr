import chainer
import chainer.functions as F
import chainer.links as L

class CNNAE3D512(chainer.Chain):

    def __init__(self, train=True):
        w = chainer.initializers.Normal(0.02)                                                           # 零均值高斯噪声，标准差为0.02
        super(CNNAE3D512, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1,initialW=w),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1,initialW=w),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1,initialW=w),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1,initialW=w),
            c4 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),
            c5 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),
            c6 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),
            c7 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),
            c8 = L.Convolution2D(512, 512, 4, stride=2, pad=1,initialW=w),

            dc00 = L.DeconvolutionND(3, 512, 512, (4, 4, 4), stride=(2,2,2), pad=1,initialW=w),
            dc0 = L.DeconvolutionND(3, 1024, 512, (4, 4, 4), stride=(2,2,2), pad=1,initialW=w),
            dc1 = L.DeconvolutionND(3, 1024, 512, (4, 4, 4), stride=(2,2,2), pad=1,initialW=w),
            dc2 = L.DeconvolutionND(3, 1024, 512, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc3 = L.DeconvolutionND(3, 1024, 512, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc4 = L.DeconvolutionND(3, 1024, 256, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc5 = L.DeconvolutionND(3, 512, 128, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc6 = L.DeconvolutionND(3, 256, 64, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),
            dc7 = L.DeconvolutionND(3, 128, 3, (3, 4, 4), stride=(1,2,2), pad=1,initialW=w),

            bnc1 = L.BatchNormalization(128),
            bnc2 = L.BatchNormalization(256),
            bnc3 = L.BatchNormalization(512),
            bnc4 = L.BatchNormalization(512),
            bnc5 = L.BatchNormalization(512),
            bnc6 = L.BatchNormalization(512),
            bnc7 = L.BatchNormalization(512),
            bnc8 = L.BatchNormalization(512),

            bndc00 = L.BatchNormalization(512),
            bndc0 = L.BatchNormalization(512),
            bndc1 = L.BatchNormalization(512),
            bndc2 = L.BatchNormalization(512),
            bndc3 = L.BatchNormalization(512),
            bndc4 = L.BatchNormalization(256),
            bndc5 = L.BatchNormalization(128),
            bndc6 = L.BatchNormalization(64)
        )
        self.train = train
        self.train_dropout = train

# origin code
    #def __call__(self, xi):
    #     hc0 = F.leaky_relu(self.c0(xi))
    #     hc1 = F.leaky_relu(self.bnc1(self.c1(hc0), test=not self.train))
    #     hc2 = F.leaky_relu(self.bnc2(self.c2(hc1), test=not self.train))
    #     hc3 = F.leaky_relu(self.bnc3(self.c3(hc2), test=not self.train))
    #     hc4 = F.leaky_relu(self.bnc4(self.c4(hc3), test=not self.train))
    #     hc5 = F.leaky_relu(self.bnc5(self.c5(hc4), test=not self.train))
    #     hc6 = F.leaky_relu(self.bnc6(self.c6(hc5), test=not self.train))
    #     hc7 = F.leaky_relu(self.bnc7(self.c7(hc6), test=not self.train))
    #     hc8 = F.leaky_relu(self.bnc8(self.c8(hc7), test=not self.train))

    #     h = F.expand_dims(hc8,2)
    #     h = F.relu(F.dropout(self.bndc00(self.dc00(h), test=not self.train), 0.5, train=self.train_dropout))
    #     hc7 = F.expand_dims(hc7,2)
    #     hc7 = F.broadcast_to(hc7, hc7.data.shape[:2]+(h.data.shape[2],)+hc7.data.shape[3:])
    #     h = F.concat((h,hc7),1)
    #     h = F.relu(F.dropout(self.bndc0(self.dc0(h), test=not self.train), 0.5, train=self.train_dropout))
    #     hc6 = F.expand_dims(hc6,2)
    #     hc6 = F.broadcast_to(hc6, hc6.data.shape[:2]+(h.data.shape[2],)+hc6.data.shape[3:])
    #     h = F.concat((h,hc6),1)
    #     h = F.relu(F.dropout(self.bndc1(self.dc1(h), test=not self.train), 0.5, train=self.train_dropout))
    #     hc5 = F.expand_dims(hc5,2)
    #     hc5 = F.broadcast_to(hc5, hc5.data.shape[:2]+(h.data.shape[2],)+hc5.data.shape[3:])
    #     h = F.concat((h,hc5),1)
    #     h = F.relu(self.bndc2(self.dc2(h), test=not self.train))
    #     hc4 = F.expand_dims(hc4,2)
    #     hc4 = F.broadcast_to(hc4, hc4.data.shape[:2]+(h.data.shape[2],)+hc4.data.shape[3:])
    #     h = F.concat((h,hc4),1)
    #     h = F.relu(self.bndc3(self.dc3(h), test=not self.train))
    #     hc3 = F.expand_dims(hc3,2)
    #     hc3 = F.broadcast_to(hc3, hc3.data.shape[:2]+(h.data.shape[2],)+hc3.data.shape[3:])
    #     h = F.concat((h,hc3),1)
    #     h = F.relu(self.bndc4(self.dc4(h), test=not self.train))
    #     hc2 = F.expand_dims(hc2,2)
    #     hc2 = F.broadcast_to(hc2, hc2.data.shape[:2]+(h.data.shape[2],)+hc2.data.shape[3:])
    #     h = F.concat((h,hc2),1)
    #     h = F.relu(self.bndc5(self.dc5(h), test=not self.train))
    #     hc1 = F.expand_dims(hc1,2)
    #     hc1 = F.broadcast_to(hc1, hc1.data.shape[:2]+(h.data.shape[2],)+hc1.data.shape[3:])
    #     h = F.concat((h,hc1),1)
    #     h = F.relu(self.bndc6(self.dc6(h), test=not self.train))
    #     hc0 = F.expand_dims(hc0,2)
    #     hc0 = F.broadcast_to(hc0, hc0.data.shape[:2]+(h.data.shape[2],)+hc0.data.shape[3:])
    #     h = F.concat((h,hc0),1)
    #     h = self.dc7(h)

    #     xi_ = F.expand_dims(xi,2)
    #     xi_ = F.broadcast_to(xi_, h.data.shape)

    #     h = F.sigmoid(h+xi_)
    #     return h


# modify code

    def __call__(self, xi):
        hc0 = F.leaky_relu(self.c0(xi))                     # size = (1,64,256,256)
        hc1 = F.leaky_relu(self.bnc1(self.c1(hc0)))         # size = (1,128,128,128)
        hc2 = F.leaky_relu(self.bnc2(self.c2(hc1)))         # size = (1,256,64,64)
        hc3 = F.leaky_relu(self.bnc3(self.c3(hc2)))         # size = (1,512,32,32)
        hc4 = F.leaky_relu(self.bnc4(self.c4(hc3)))         # size = (1,512,16,16)
        hc5 = F.leaky_relu(self.bnc5(self.c5(hc4)))         # size = (1,512,8,8)
        hc6 = F.leaky_relu(self.bnc6(self.c6(hc5)))         # size = (1,512,4,4)
        hc7 = F.leaky_relu(self.bnc7(self.c7(hc6)))         # size = (1,512,2,2)
        hc8 = F.leaky_relu(self.bnc8(self.c8(hc7)))         # size = (1,512,1,1)

        h = F.expand_dims(hc8,2)                                                                                                     # size = (1,512,1,1,1)
        h = F.relu(F.dropout(self.bndc00(self.dc00(h), ), 0.5))                                                         # size = (1,512,2,2,2)  0.5 dropout
        hc7 = F.expand_dims(hc7,2)                                                                                                   # size = (1,512,1,2,2)
        hc7 = F.broadcast_to(hc7, hc7.data.shape[:2]+(h.data.shape[2],)+hc7.data.shape[3:])                                                # size = (1,512,2,2,2)
        h = F.concat((h,hc7),1)                                                                                                                # size = (1,1024,2,2,2)
        h = F.relu(F.dropout(self.bndc0(self.dc0(h), ), 0.5))                                                                  # size = (1,512,4,4,4)
        hc6 = F.expand_dims(hc6,2)                                                                                                         # size = (1,512,1,4,4)
        hc6 = F.broadcast_to(hc6, hc6.data.shape[:2]+(h.data.shape[2],)+hc6.data.shape[3:])                                              # size = (1,512,4,4,4)
        h = F.concat((h,hc6),1)                                                                                                             # size = (1,1024,4,4,4)
        h = F.relu(F.dropout(self.bndc1(self.dc1(h), ), 0.5))                                                                      # size = (1,512,8,8,8)
        hc5 = F.expand_dims(hc5,2)                                                                                                        # size = (1,512,1,8,8)
        hc5 = F.broadcast_to(hc5, hc5.data.shape[:2]+(h.data.shape[2],)+hc5.data.shape[3:])                                               # size = (1,512,8,8,8)
        h = F.concat((h,hc5),1)                                                                                                                 # size = (1,1024,8,8,8)
        h = F.relu(self.bndc2(self.dc2(h)))                                                                                                # size = (1,512,8,16,16)
        hc4 = F.expand_dims(hc4,2)                                                                                                      # size = (1,512,1,16,16)
        hc4 = F.broadcast_to(hc4, hc4.data.shape[:2]+(h.data.shape[2],)+hc4.data.shape[3:])                                          # size = (1,512,8,16,16)
        h = F.concat((h,hc4),1)                                                                                                                # size = (1,1024,8,16,16)
        h = F.relu(self.bndc3(self.dc3(h)))                                                                                              # size = (1,512,8,32,32)
        hc3 = F.expand_dims(hc3,2)                                                                                                      # size = (1,512,1,32,32)
        hc3 = F.broadcast_to(hc3, hc3.data.shape[:2]+(h.data.shape[2],)+hc3.data.shape[3:])                                           # size = (1,512,8,32,32)
        h = F.concat((h,hc3),1)                                                                                                                  # size = (1,1024,8,32,32)
        h = F.relu(self.bndc4(self.dc4(h)))                                                                                           # size = (1,256,8,64,64)
        hc2 = F.expand_dims(hc2,2)                                                                                                     # size = (1,256,1,64,64)
        hc2 = F.broadcast_to(hc2, hc2.data.shape[:2]+(h.data.shape[2],)+hc2.data.shape[3:])                                           # size = (1,256,8,64,64)
        h = F.concat((h,hc2),1)                                                                                                                # size = (1,512,8,64,64)
        h = F.relu(self.bndc5(self.dc5(h)))                                                                                               # size = (1,128,8,128,128)
        hc1 = F.expand_dims(hc1,2)                                                                                                      # size = (1,128,1,128,128)
        hc1 = F.broadcast_to(hc1, hc1.data.shape[:2]+(h.data.shape[2],)+hc1.data.shape[3:])                                           # size = (1,128,8,128,128)
        h = F.concat((h,hc1),1)                                                                                                             # size = (1,256,8,128,128)
        h = F.relu(self.bndc6(self.dc6(h)))                                                                                           # size = (1,64,8,256,256)
        hc0 = F.expand_dims(hc0,2)                                                                                                   # size = (1,64,1,256,256)
        hc0 = F.broadcast_to(hc0, hc0.data.shape[:2]+(h.data.shape[2],)+hc0.data.shape[3:])                                      # size = (1,64,8,256,256)
        h = F.concat((h,hc0),1)                                                                                                            # size = (1,128,8,256,256)
        h = self.dc7(h)                                                                                                                        # size = (1,3,8,512,512)

        xi_ = F.expand_dims(xi,2)                                            # size = (1,3,1,512,512)
        xi_ = F.broadcast_to(xi_, h.data.shape)                       # size = (1,3,8,512,512)

        h = F.sigmoid(h+xi_)                                                    # size = (1,3,8,512,512)
        return h