import torch
torch.cuda.empty_cache()

import sys
sys.exit(__import__('test').main())
