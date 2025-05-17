import sys
sys.path.append("./model")

from internvl3.internvl3_embedder import InternVL3Embedder

def main():
    embedder = InternVL3Embedder()

    # 🔹 测试 chat 功能（图文对话）
    print("=== Chat Demo ===")
    response = embedder.chat("/home/bozhao/code/miravla/model/internvl3/example.jpg", "请描述图中物体")
    print("Assistant:", response)

    # 🔹 测试图像嵌入（用于 GR00T/RDT 融合）
    # print("\n=== Embedding Demo ===")
    # image_embed = embedder.get_image_embedding("model/internvl3/example.jpg")
    # print("Image Embedding Shape:", image_embed.shape)

if __name__ == "__main__":
    main()
