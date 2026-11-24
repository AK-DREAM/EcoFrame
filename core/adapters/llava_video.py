from core.adapters.llava_vid_attention import LlavaVideoTracker
from core.adapters.llava_onevision import LlavaOneVisionCToFAdapter


class LlavaVideoCToFAdapter(LlavaOneVisionCToFAdapter):
    tracker_cls = LlavaVideoTracker
