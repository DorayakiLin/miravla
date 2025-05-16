# scripts/test_infer.py
import sys
sys.path.append("./model")  # 添加 model 路径

from internvl3.internvl3_embedder import InternVL3Embedder

def main():
    model = InternVL3Embedder()
    embedding = model.get_embedding("model/internvl3/example.jpg", "describe the object on the table")

    # 示例：打印 CLS token embedding 的形状
    cls_token = embedding[:, 0, :]
    print("CLS token embedding shape:", cls_token.shape)

if __name__ == "__main__":
    main()
