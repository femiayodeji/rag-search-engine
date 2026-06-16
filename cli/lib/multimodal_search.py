from PIL import Image
from sentence_transformers import SentenceTransformer
from cli.lib.semantic_search import cosine_similarity
from cli.load_data import get_movies

class MultimodalSearch:
    def __init__(self, documents: list, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts: list[str] = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        return self.model.encode([image])[0]
    
    def search_with_image(self, image_path: str):
        image_embedding = self.embed_image(image_path)
        similarities: list[dict] = [ {"text": text, "similarity": cosine_similarity(image_embedding, text_embedding)} for text, text_embedding in zip(self.texts, self.text_embeddings)]
        ranked_matches = sorted(range(len(similarities)), key=lambda i: similarities[i]["similarity"], reverse=True)[:5]
        results = []
        for ranked_match in ranked_matches:
            results.append({
                "document_id": self.documents[ranked_match]["id"],
                "title": self.documents[ranked_match]["title"],
                "description": self.documents[ranked_match]["description"],
                "similarity": similarities[ranked_match]["similarity"]
            })
        return results

def verify_image_embedding(image_path: str):
    search = MultimodalSearch(documents=[])
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> list[dict]:
    movies = get_movies()
    search = MultimodalSearch(documents=movies)
    return search.search_with_image(image_path)