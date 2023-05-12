# import ot
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
import multiprocessing
import itertools

# from qtorch import FixedPoint
# from qtorch.optim import OptimLP
# from qtorch.quant import Quantizer, fixed_point_quantize
from collections import OrderedDict
from .telescope_pt import telescopeMSE8x8, move_constants_to_gpu
from .autoencoder_datamodule import ARRANGE, ARRANGE_MASK


"""
Model: "encoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 8, 8, 1)]         0         
                                                                 
 input_qa (QActivation)      (None, 8, 8, 1)           0         
                                                                 
 conv2d_0_m (FQConv2D)       (None, 4, 4, 8)           80        
                                                                 
 accum1_qa (QActivation)     (None, 4, 4, 8)           0         
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 encoded_vector (FQDense)    (None, 16)                2064      
                                                                 
 encod_qa (QActivation)      (None, 16)                0         
                                                                 
=================================================================
Total params: 2,144
Trainable params: 2,144
Non-trainable params: 0

_________________________________________________________________
Model: "decoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 decoder_input (InputLayer)  [(None, 16)]              0         
                                                                 
 de_dense_final (Dense)      (None, 128)               2176      
                                                                 
 de_reshape (Reshape)        (None, 4, 4, 8)           0         
                                                                 
 conv2D_t_0 (Conv2DTranspose  (None, 8, 8, 8)          584       
 )                                                               
                                                                 
 conv2d_t_final (Conv2DTrans  (None, 8, 8, 1)          73        
 pose)                                                           
                                                                 
 decoder_output (Activation)  (None, 8, 8, 1)          0         
                                                                 
=================================================================
Total params: 2,833
Trainable params: 2,833
Non-trainable params: 0
"""


CALQ_MASK = torch.tensor(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
    ]
)


class AutoEncoder(pl.LightningModule):
    """
    XDR AutoEncoder class
    """

    def __init__(self, accelerator="gpu", quantize=False) -> None:
        super().__init__()

        self.encoded_dim = 16
        self.shape = (1, 8, 8)  # PyTorch defaults to (C, H, W)
        self.val_sum = None
        self.accelerator = accelerator
        self.quantize = quantize

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)),
                    ("relu", nn.ReLU()),
                    ("flatten", nn.Flatten()),
                    ("enc_dense", nn.Linear(128, self.encoded_dim)),
                    ("relu1", nn.ReLU()),
                ]
            )
        )
        if self.quantize:
            self.automatic_optimization = False
            input_forward_num = FixedPoint(wl=4, fl=7) # 11-bit input
            input_quantizer = Quantizer(forward_number=input_forward_num, forward_rounding="stochastic")
            accum_forward_num = FixedPoint(wl=4, fl=8) # 12-bit accumulator
            accum_quantizer = Quantizer(forward_number=accum_forward_num, forward_rounding="stochastic")
            encoder_out_forward_num = FixedPoint(wl=2, fl=8) # 10-bit encoder output
            encoder_out_quantizer = Quantizer(
                forward_number=encoder_out_forward_num, 
                forward_rounding="stochastic"
            )
            self.encoder = nn.Sequential(OrderedDict([
                ("input_quant", input_quantizer),
                ("qconv", nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)),
                ("accum_quant1", accum_quantizer),
                ("relu", nn.ReLU()),
                ("accum_quant2", accum_quantizer),
                ("flatten", nn.Flatten()),
                ("qenc_dense", nn.Linear(128, self.encoded_dim)),
                ("encoder_out_quant1", encoder_out_quantizer),
                ("relu1", nn.ReLU()),
                ("encoder_out_quant2", encoder_out_quantizer),
            ]))

            self.weight_quantizer = lambda x : fixed_point_quantize(x, wl=2, fl=4, rounding="stochastic")

        self.decoder = nn.Sequential(OrderedDict([
            ("dec_dense", nn.Linear(self.encoded_dim, 128)),
            ("relu1", nn.ReLU()),
            ("unflatten", nn.Unflatten(1, (8, 4, 4))),
            (
                "convtrans2d1",
                nn.ConvTranspose2d(
                    8, 8, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
            ),
            ("relu2", nn.ReLU()),
            (
                "convtrans2d2",
                nn.ConvTranspose2d(
                    8,
                    self.shape[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ),
            ("sigmoid", nn.Sigmoid()),
        ]))
        self.loss = telescopeMSE8x8
        if accelerator == "gpu":
            print("Moved constants to gpu")
            move_constants_to_gpu()

    def invert_arrange(self):
        """
        Invert the arrange mask
        """
        remap = []
        hashmap = {}  # cell : index mapping
        found_duplicate_charge = len(ARRANGE[ARRANGE_MASK == 1]) > len(
            torch.unique(ARRANGE[ARRANGE_MASK == 1])
        )
        for i in range(len(ARRANGE)):
            if ARRANGE_MASK[i] == 1:
                if found_duplicate_charge:
                    if CALQ_MASK[i] == 1:
                        hashmap[int(ARRANGE[i])] = i
                else:
                    hashmap[int(ARRANGE[i])] = i
        for i in range(len(torch.unique(ARRANGE))):
            remap.append(hashmap[i])
        return torch.tensor(remap)

    def map_to_calq(self, x):
        """
        Map the input/output of the autoencoder into CALQs orders
        """
        remap = self.invert_arrange()
        image_size = self.shape[0] * self.shape[1] * self.shape[2]
        reshaped_x = x.reshape(len(x), image_size)
        reshaped_x[:, ARRANGE_MASK == 0] = 0
        return reshaped_x[:, remap]

    def set_val_sum(self, val_sum):
        self.val_sum = val_sum

    def predict(self, x):
        decoded_Q = self(x)
        encoded_Q = self.encoder(x)
        encoded_Q = torch.reshape(encoded_Q, (len(encoded_Q), self.encoded_dim, 1))
        return decoded_Q, encoded_Q

    # Pytorch Lightning specific methods
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        if self.quantize:
            optimizer = OptimLP(
                optimizer, 
                weight_quant=self.weight_quantizer,
                grad_quant=self.weight_quantizer,
                momentum_quant=self.weight_quantizer,
            )
        return optimizer

    def training_step(self, batch, batch_idx):

        x = batch
        x_hat = self(x)
        loss = self.loss(x, x_hat)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        if self.quantize:
            # Automatic optimization is disabled for quantization
            # so we need to manually call the optimizer
            self.manual_backward(loss)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss(x, x_hat)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
