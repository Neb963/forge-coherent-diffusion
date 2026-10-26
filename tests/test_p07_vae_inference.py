"""Real PyTorch inference tensor regression mirroring Forge tiled VAE.

Forge's tiled_scale_multidim is decorated @torch.inference_mode(), returning
an inference tensor; process_output calls add_/div_/clamp_ after that helper
returns. Without enclosing the native decode, the exact installed-host
RuntimeError is raised.
"""
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"Requires PyTorch inference tensor semantics")
class NativeTiledDecoderContracts(unittest.TestCase):
    def fake_vae(self, wan):
        class FakeVAE:
            first_stage_model=types.SimpleNamespace(process_out=lambda z:z)
            def decode_tiled(self,samples,*,tile_x,tile_y,overlap):
                # Reproduce Forge's @inference_mode on tiled_scale_multidim.
                @torch.inference_mode()
                def tiled_scale_multidim():
                    shape=(1,1,8,8,3) if wan else (1,8,8,3)
                    return torch.zeros(shape,dtype=torch.float32)
                image=tiled_scale_multidim()
                self.created_inference=image.is_inference()
                # Actual Forge VAE.process_output implementation.
                return image.add_(1.0).div_(2.0).clamp_(0.0,1.0)
        return FakeVAE()

    def test_unwrapped_native_postprocessor_reproduces_real_failure(self):
        vae=self.fake_vae(True)
        with self.assertRaisesRegex(RuntimeError,"Inplace update to inference tensor"):
            vae.decode_tiled(None,tile_x=32,tile_y=32,overlap=8)

    def test_anima_native_wan_tiled_decode_completes_after_fix(self):
        from coherent_diffusion.forge_bridge.anima import ForgeAnima
        vae=self.fake_vae(True)
        host=types.SimpleNamespace(forge_objects=types.SimpleNamespace(vae=vae))
        adapter=ForgeAnima(host,None,None,None,1.0,1,"cpu",2)
        z=torch.zeros((1,16,1,1,1),dtype=torch.float32)
        with patch.object(ForgeAnima,"check_host",return_value=2):
            output=adapter.decode_tiled(z)
        self.assertTrue(vae.created_inference)
        self.assertEqual(tuple(output.shape),(1,8,8,3))
        self.assertEqual(float(output.mean()),.5)

    def test_sdxl_native_tiled_decoder_completes_after_fix(self):
        from coherent_diffusion.forge_bridge.sdxl import ForgeSDXL
        vae=self.fake_vae(False)
        host=types.SimpleNamespace(forge_objects=types.SimpleNamespace(vae=vae))
        adapter=ForgeSDXL(host,None,None,None,1.0,1,"cpu")
        z=torch.zeros((1,4,1,1),dtype=torch.float32)
        with patch.object(ForgeSDXL,"_check_host"):
            output=adapter.decode_tiled(z)
        self.assertTrue(vae.created_inference)
        self.assertEqual(tuple(output.shape),(1,8,8,3))
        self.assertEqual(float(output.mean()),.5)

    def test_decoder_failure_preserves_native_exception(self):
        from coherent_diffusion.forge_bridge.anima import ForgeAnima
        class Exploding:
            first_stage_model=types.SimpleNamespace(process_out=lambda x:x)
            def decode_tiled(self,*args,**kwargs):
                raise RuntimeError("native VAE decode failed")
        host=types.SimpleNamespace(forge_objects=types.SimpleNamespace(vae=Exploding()))
        adapter=ForgeAnima(host,None,None,None,1.0,1,"cpu",2)
        with patch.object(ForgeAnima,"check_host",return_value=2),\
             self.assertRaisesRegex(RuntimeError,"native VAE decode failed"):
            adapter.decode_tiled(torch.zeros((1,16,1,1,1)))
