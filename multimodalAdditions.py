import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape

        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class VisionAttention(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.heads = heads
        self.head_dims = dim//heads

        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B,T,C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dims).transpose(1,2)
        k = k.view(B, T, self.heads, self.head_dims).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dims).transpose(1, 2)

        att = q @ k.transpose(-2, -1) / (self.head_dims) ** 0.5
        
        att = torch.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1,2).reshape(B, T, C)

        return self.proj(out)

class ViTBlock(nn.Module):
    def __init__(self, dim=768):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = VisionAttention(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
                                    nn.Linear(dim, 4*dim),
                                    nn.GELU(),
                                    nn.Linear(4*dim, dim)
                                )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, layers=6, dim=768):
        super().__init__()

        self.patch = PatchEmbedding()

        self.blocks = nn.ModuleList([ViTBlock(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, image):
        x = self.patch(image)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
class VisionAdapter(nn.Module):
    def __init__(self, vision_dim=768, llm_dim=4096):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim=4096, heads=32):

        super().__init__()
        self.heads = heads
        self.head_dim = dim//heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, text, image):
        B,T,C = text.shape
        _,I,_ = image.shape

        q = self.q(text)
        k = self.k(image)
        v = self.v(image)
        
        q = q.view(B, T, self.heads, self.head_dim).transpose(1,2)
        k = k.view(B, I, self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, I, self.heads, self.head_dim).transpose(1,2)

        attn = (q @ k.transpose(-2,-1))/(self.head_dim ** 0.5)

        attn = F.softmax(attn, dim=-1)

        out = attn @ v

        out = out.transpose(1,2).reshape(B,T,C)
        return self.proj(out)

class CasualSelfAttention(nn.Module):
    def __init__(self, dim=4096, heads=32, block_size=2048 ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)


        q = q.view(B,T,self.heads, self.head_dim).transpose(1,2)
        k = k.view(B,T,self.heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1,2)

        attn = (q @ k.transpose(-1,-2)) / (self.head_dim ** 0.5)

        mask = self.mask[:T, :T]

        attn = attn.masked_fill(mask==0, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)

        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x



class MultiModalBlock(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CasualSelfAttention(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
    
    def forward(self, text, image):
        
        text = text + self.self_attn(self.norm1(text))

        text = text + self.cross_attn(self.norm2(text), image)

        text = text + self.mlp(self.norm3(text))

        return text

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, tokens):
        return self.embed(tokens)


class GPT4Vision(nn.Module):
    def __init__(self, vocab_size, dim=4096, layers=24):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, dim)
        
        self.vision = VisionTransformer()

        self.adapter = VisionAdapter(768, dim)

        self.blocks = nn.ModuleList([MultiModalBlock(dim) for _ in range(layers)])

        self.norm = nn.LayerNorm(dim)

        self. lm_head = nn.Linear(dim, vocab_size)


    def forward(self, image, tokens):
        image_tokens = self.vision(image)
        image_tokens = self.adapter(image_tokens)

        text = self.token_embed(tokens)

        for block in self.blocks:
            text = block(text, image_tokens)

        text = self.norm(text)

        logits = self.lm_head(text)

        return logits

class CLIPAlignment(nn.Module):
    def __init__(self, vision_dim=768, text_dim=4096, proj_dim=512):
        super().__init__()
        self.image_proj = nn.Linear(vision_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, image_emb, text_emb):
        image_emb = self.image_proj(image_emb)
        text_emb = self.text_proj(text_emb)

        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        sim = image_emb @ text_emb.T
        sim = sim / self.temperature

        return sim 

    def clip_loss(self, sim):
        labels = torch.arange(sim.size(0)).to(sim.device)

        loss_i = F.cross_entropy(sim, labels)
        loss_t = F.cross_entropy(sim.T, labels)

        return (loss_i + loss_t)/2


class vectorStore:
    def __init__(self):
        self.embeddings = []
        self.documents = []
    
    def add(self, embedding, document):
        self.embeddings.append(embedding)
        self.documents.append(document)

    def search(self, query_embedding, k=3):
        scores = []

        for emb in self.embeddings:
            score = torch.dot(query_embedding, emb)
            scores.append(score.item())

        indices = sorted( range(len(scores)), key = lambda i : scores[i], reverse=True)

        results = [self.documents[i] for i in indices[:k]]
        return results

class retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query_embedding):
        docs = self.vector_store.search(query_embedding)
        return docs

class Tools:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def run(self, input):
        return self.func(input)

    def search_tool(query):
        return retriever.retrieve(query)

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, tool):
        self.tools[tool.name] = tool

    def get(self, name):
        return self.tools[name]

class AgentMemory:
    def __init__(self):
        self.steps = []
    
    def add(self, step):
        self.steps.append(step)
    
    def context(self):
        return '\n'.join(self.steps)

def parse_action(output):
    """Parse model output to extract action and tool input"""
    lines = output.split('\n')
    action = ""
    tool_input = ""
    for line in lines:
        if "ACTION:" in line:
            action = line.split("ACTION:")[1].strip()
        elif "ACTION_INPUT:" in line:
            tool_input = line.split("ACTION_INPUT:")[1].strip()
    return action, tool_input

class Agent:

    def __init__(self, model, tool_registry):
        self.model = model
        self.tools = tool_registry
        self.memory = AgentMemory()

    def run(self, query):
        while True:
            context = self.memory.context()
            prompt = context + "\nUser:" + query

            output = self.model.generate(prompt)

            print("MODEL OUTPUT:\n", output)

            if "FINAL ANSWER" in output:
                return output

            action, tool_input = parse_action(output)

            tool = self.tools.get(action)

            if tool is None:
                return "Tool not found"

            observation = tool.run(tool_input)

            self.memory.add(output)
            self.memory.add("Observation:" + str(observation))
        











    



