import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import torch.nn.functional as F
from PIL import Image

class NomicEncoder:
    """
    Encoder for Nomic-AI's text and vision models.
    - Text model: nomic-ai/nomic-embed-text-v1.5
    - Vision model: nomic-ai/nomic-embed-vision-v1.5
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Text Model
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        self.text_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        ).to(self.device)
        self.text_model.eval()

        # Load Vision Model (correct repo identifier)
        self.vision_processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        )
        self.vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        ).to(self.device)
        self.vision_model.eval()

    @torch.no_grad()
    def encode_text(self, text, task_type="search_query"):
        """
        Encodes either a single string or a list of strings.
        """
        # Batch input
        if isinstance(text, (list, tuple)):
            if not all(isinstance(t, str) for t in text):
                raise TypeError("All items in text list must be strings.")
            prefixed = [f"{task_type}: {t}" for t in text]
            inputs = self.text_tokenizer(
                prefixed, return_tensors='pt', padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            outputs = self.text_model(**inputs, return_dict=True).last_hidden_state
            embeddings = F.normalize(outputs[:,0], p=2, dim=1)
            return embeddings

        # Single string input
        if not isinstance(text, str):
            raise TypeError("Input text must be a string or a list of strings.")
        prefixed_text = f"{task_type}: {text}"
        inputs = self.text_tokenizer(
            prefixed_text, return_tensors='pt', padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        outputs = self.text_model(**inputs, return_dict=True).last_hidden_state
        embeddings = F.normalize(outputs[:,0], p=2, dim=1)
        return embeddings

    @torch.no_grad()
    def encode_image(self, images: list):
        """
        Encodes a list of PIL images or image paths.
        """
        inputs = self.vision_processor(images, return_tensors="pt").to(self.device)
        embeddings = self.vision_model(**inputs).pooled_output
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def encode_text_and_image(self, text_image_pairs):
        """
        输入: [(text1, image1), (text2, image2), ...]
        输出: 每对组合的 embedding, numpy array
        """
        data = []
        modalities = []
        for text, image in text_image_pairs:
            data.extend([text, image])
            modalities.extend(["text", "image"])
        outputs = self.model.embed(data=data, modality=modalities)
        return np.array(outputs["embeddings"])




