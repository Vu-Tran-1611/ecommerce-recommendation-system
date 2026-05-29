from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors

output_path = "pytorch_core_functions_notes.pdf"
doc = SimpleDocTemplate(
    output_path,
    pagesize=letter,
    rightMargin=48,
    leftMargin=48,
    topMargin=48,
    bottomMargin=48,
)

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f3c88"),
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#0f172a"),
    )
)
styles.add(
    ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        leading=12,
        fontSize=9.5,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="MonoSmall",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=10.5,
        backColor=colors.whitesmoke,
        borderPadding=6,
        spaceAfter=8,
    )
)

story = []
story.append(Paragraph("PyTorch Core Functions Notes", styles["TitleCenter"]))
story.append(Spacer(1, 0.15 * inch))
story.append(
    Paragraph(
        "Focused on the functions used across Bert4Rec, ComiRec, SASRec, LightGCN, and TwoTower notebooks.",
        styles["BodySmall"],
    )
)
story.append(Spacer(1, 0.12 * inch))

sections = [
    (
        "Tensor creation and indexing",
        [
            (
                "torch.tensor(data, dtype=...)",
                "Input: Python data such as list, scalar, or NumPy array. Output: a Tensor. Purpose: create inputs, labels, masks, and IDs.",
                "x = torch.tensor([1, 2, 3], dtype=torch.long)",
            ),
            (
                "torch.arange(start, end)",
                "Input: integer range. Output: 1D Tensor. Purpose: generate positions or label indices.",
                "pos = torch.arange(5)\n# output: tensor([0, 1, 2, 3, 4])",
            ),
            (
                "torch.zeros(shape) / torch.ones(shape)",
                "Input: tensor shape. Output: filled Tensor. Purpose: build masks or placeholders.",
                "mask = torch.zeros(3, 3)",
            ),
            (
                "tensor[idx]",
                "Input: integer/slice/mask. Output: selected values. Purpose: read embeddings, logits, or scores.",
                "first_row = x[0]",
            ),
        ],
    ),
    (
        "Shape and layout control",
        [
            (
                "Tensor.unsqueeze(dim)",
                "Input: Tensor and a dimension. Output: Tensor with one extra axis. Purpose: add batch or sequence dimensions.",
                "x = torch.tensor([1, 2, 3]).unsqueeze(0)",
            ),
            (
                "Tensor.squeeze(dim)",
                "Input: Tensor. Output: Tensor with size-1 dimensions removed. Purpose: remove fake batch dimensions after inference.",
                "x = torch.randn(1, 6, 64).squeeze(0)",
            ),
            (
                "Tensor.expand(...)",
                "Input: Tensor with size-1 dimensions plus target sizes. Output: a broadcasted view. Purpose: repeat values without copying data.",
                "x = torch.arange(4).unsqueeze(0).expand(2, -1)\n# output shape: torch.Size([2, 4])\n# output: tensor([[0, 1, 2, 3],\n#                 [0, 1, 2, 3]])",
            ),
            (
                "Tensor.repeat(...)",
                "Input: Tensor and repeat counts. Output: physically repeated Tensor. Purpose: duplicate values when broadcasting is not enough.",
                "x = torch.tensor([[1, 2]])\ny = x.repeat(3, 1)\n# output: tensor([[1, 2],\n#                 [1, 2],\n#                 [1, 2]])",
            ),
            (
                "Tensor.view(...)",
                "Input: Tensor and new shape. Output: reshaped view if layout is compatible. Purpose: fast reshape for contiguous tensors.",
                "x = torch.arange(6).view(2, 3)\n# output: tensor([[0, 1, 2],\n#                 [3, 4, 5]])",
            ),
            (
                "Tensor.reshape(...)",
                "Input: Tensor and new shape. Output: reshaped Tensor, copying if needed. Purpose: safer reshape when contiguity is unknown.",
                "x = torch.arange(6).reshape(2, 3)\n# output: tensor([[0, 1, 2],\n#                 [3, 4, 5]])",
            ),
            (
                "Tensor.transpose(dim0, dim1)",
                "Input: two dimensions. Output: Tensor with those axes swapped. Purpose: switch layout for attention or matrix ops.",
                "x = x.transpose(1, 2)",
            ),
            (
                "Tensor.permute(...)",
                "Input: Tensor and a full dimension order. Output: Tensor with reordered axes. Purpose: flexible axis rearrangement.",
                "x = x.permute(0, 2, 1)",
            ),
            (
                "Tensor.t()",
                "Input: 2D Tensor. Output: transposed matrix. Purpose: shorthand for matrix transpose.",
                "m = torch.tensor([[1, 2], [3, 4]]).t()",
            ),
            (
                "expand vs repeat",
                "expand broadcasts without copying; repeat copies data. Purpose: use expand for cheap broadcasting, repeat for actual duplication.",
                "x = torch.tensor([[1, 2, 3]])\na = x.expand(4, -1)\n# output: tensor([[1, 2, 3],\n#                 [1, 2, 3],\n#                 [1, 2, 3],\n#                 [1, 2, 3]])\nb = x.repeat(4, 1)\n# output: tensor([[1, 2, 3],\n#                 [1, 2, 3],\n#                 [1, 2, 3],\n#                 [1, 2, 3]])",
            ),
            (
                "view vs reshape",
                "view only works for compatible memory layout; reshape is more flexible. Purpose: use reshape by default if layout may be non-contiguous.",
                "x = torch.arange(6).reshape(2, 3)\nx_t = x.transpose(0, 1)\n# x_t output: tensor([[0, 3],\n#                     [1, 4],\n#                     [2, 5]])\n# x_t.view(...) may fail if memory is not contiguous\n# x_t.reshape(3, 2) output: tensor([[0, 3],\n#                                  [1, 4],\n#                                  [2, 5]])",
            ),
        ],
    ),
    (
        "Masking and attention utilities",
        [
            (
                "torch.triu(tensor, diagonal=1)",
                "Input: square Tensor. Output: upper triangle. Purpose: causal attention mask so a token cannot see future tokens.",
                "mask = torch.triu(torch.ones(4, 4), diagonal=1)",
            ),
            (
                "Tensor.eq(value)",
                "Input: Tensor and value. Output: boolean Tensor. Purpose: find padding positions.",
                "padding_mask = input_ids.eq(0)",
            ),
            (
                "Tensor.masked_fill(mask, value)",
                "Input: boolean mask and fill value. Output: Tensor with selected entries replaced. Purpose: hide padding or future positions.",
                "scores = scores.masked_fill(mask, float(\"-inf\"))",
            ),
        ],
    ),
    (
        "Math and ranking",
        [
            (
                "torch.sum(x, dim=...)",
                "Input: Tensor. Output: reduced Tensor. Purpose: dot-product style scoring and reductions.",
                "score = torch.sum(user_emb * item_emb, dim=1)",
            ),
            (
                "torch.dot(a, b)",
                "Input: two 1D Tensors. Output: scalar. Purpose: similarity score between embeddings.",
                "score = torch.dot(user_vec, item_vec)",
            ),
            (
                "torch.matmul(a, b) / @",
                "Input: compatible tensors. Output: matrix product. Purpose: score all items at once.",
                "scores = user_vec @ item_vec.T",
            ),
            (
                "torch.bmm(a, b)",
                "Input: two 3D Tensors with batch dimension. Output: batched matrix multiply. Purpose: ComiRec interest-to-item scoring.",
                "out = torch.bmm(A, H)",
            ),
            (
                "torch.max(x, dim=...)",
                "Input: Tensor. Output: values and indices of maxima. Purpose: pick the best interest or best score.",
                "values, idx = torch.max(scores_by_interest, dim=0)",
            ),
            (
                "torch.argmax(x, dim=...)",
                "Input: Tensor. Output: index of largest value. Purpose: select the best matching interest vector.",
                "best_idx = torch.argmax(interest_scores, dim=-1)",
            ),
            (
                "torch.topk(x, k)",
                "Input: Tensor and k. Output: top values and indices. Purpose: get top recommendations.",
                "vals, idx = torch.topk(scores, k=10)",
            ),
            (
                "torch.nan_to_num(x, nan=0.0)",
                "Input: Tensor. Output: Tensor with NaNs replaced. Purpose: stabilize outputs when training/debugging.",
                "x = torch.nan_to_num(x, nan=0.0)",
            ),
        ],
    ),
    (
        "Neural network layers",
        [
            (
                "nn.Module",
                "Base class for all trainable PyTorch models. Purpose: define custom models with parameters and forward pass.",
                "class MyModel(nn.Module): ...",
            ),
            (
                "nn.Embedding(num_embeddings, embedding_dim, padding_idx=...)",
                "Input: integer IDs. Output: dense learned vectors. Purpose: user/item/category/position embeddings.",
                "emb = nn.Embedding(1000, 64, padding_idx=0)",
            ),
            (
                "nn.Linear(in_features, out_features)",
                "Input: feature vectors. Output: affine-transformed vectors. Purpose: projections and output heads.",
                "proj = nn.Linear(256, 64)",
            ),
            (
                "nn.Dropout(p)",
                "Input: Tensor. Output: Tensor with random activations dropped during training. Purpose: regularization.",
                "drop = nn.Dropout(0.2)",
            ),
            (
                "nn.LayerNorm(normalized_shape)",
                "Input: Tensor. Output: normalized Tensor. Purpose: stabilize transformer activations.",
                "norm = nn.LayerNorm(64)",
            ),
            (
                "nn.ReLU()",
                "Input: Tensor. Output: Tensor with negatives clipped to zero. Purpose: nonlinearity in MLPs.",
                "act = nn.ReLU()",
            ),
            (
                "nn.TransformerEncoderLayer(...)",
                "Input: sequence embeddings. Output: transformed sequence embeddings. Purpose: one transformer block in SASRec/Bert4Rec.",
                "layer = nn.TransformerEncoderLayer(d_model=64, nhead=2, batch_first=True)",
            ),
            (
                "nn.TransformerEncoder(...)",
                "Input: a transformer layer plus layer count. Output: stacked transformer encoder. Purpose: build multiple blocks.",
                "encoder = nn.TransformerEncoder(layer, num_layers=2)",
            ),
        ],
    ),
    (
        "Functional API and training helpers",
        [
            (
                "F.softmax(x, dim=...)",
                "Input: scores/logits. Output: probabilities that sum to 1. Purpose: attention weights in ComiRec.",
                "weights = F.softmax(scores, dim=-1)",
            ),
            (
                "F.normalize(x, dim=...)",
                "Input: vector Tensor. Output: unit-length vectors. Purpose: cosine-like similarity in TwoTower.",
                "v = F.normalize(v, dim=1)",
            ),
            (
                "F.cross_entropy(logits, targets)",
                "Input: raw logits and class labels. Output: scalar loss. Purpose: classification/retrieval training.",
                "loss = F.cross_entropy(logits, labels)",
            ),
            (
                "torch.no_grad()",
                "Context manager. Output: disables gradient tracking. Purpose: faster inference and less memory use.",
                "with torch.no_grad(): scores = model(x)",
            ),
            (
                "torch.manual_seed(seed)",
                "Input: integer seed. Output: none. Purpose: reproducible initialization and sampling.",
                "torch.manual_seed(42)",
            ),
            (
                "torch.device(...)",
                "Input: device string. Output: device object. Purpose: choose CPU, CUDA, or MPS.",
                "device = torch.device(\"mps\")",
            ),
            (
                "tensor.to(device)",
                "Input: Tensor or device/dtype. Output: moved Tensor. Purpose: place data on the right hardware.",
                "x = x.to(device)",
            ),
            (
                "torch.save(obj, path)",
                "Input: model state, tensor, or Python object. Output: file on disk. Purpose: save checkpoints and mappings.",
                "torch.save(model.state_dict(), \"model.pth\")",
            ),
            (
                "torch.load(path)",
                "Input: saved file. Output: restored object. Purpose: reload checkpoints, embeddings, mappings.",
                "ckpt = torch.load(\"model.pth\")",
            ),
            (
                "optimizer.zero_grad()",
                "Input: none. Output: clears gradients. Purpose: reset gradients before backprop.",
                "optimizer.zero_grad()",
            ),
            (
                "loss.backward()",
                "Input: scalar loss. Output: fills .grad fields. Purpose: compute gradients.",
                "loss.backward()",
            ),
            (
                "optimizer.step()",
                "Input: none. Output: updates weights. Purpose: apply gradient descent/Adam updates.",
                "optimizer.step()",
            ),
        ],
    ),
    (
        "Data loading",
        [
            (
                "Dataset",
                "Custom class with __len__ and __getitem__. Output: dataset object. Purpose: define sample loading logic.",
                "class MyDataset(Dataset): ...",
            ),
            (
                "DataLoader",
                "Input: dataset, batch size, shuffle flag. Output: iterable of mini-batches. Purpose: efficient batching.",
                "loader = DataLoader(dataset, batch_size=64, shuffle=True)",
            ),
        ],
    ),
    (
        "Common notebook patterns",
        [
            (
                "SASRec / Bert4Rec",
                "Used arange, unsqueeze, expand, triu, masked_fill, TransformerEncoderLayer, no_grad for sequential attention and inference.",
                "positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)",
            ),
            (
                "ComiRec",
                "Used stack, transpose, softmax, bmm, max to build multi-interest user vectors and score items.",
                "price_features = torch.stack([price_norm, offer_price_norm], dim=1)",
            ),
            (
                "TwoTower",
                "Used Embedding, Linear, cat, stack, normalize, matmul, cross_entropy for dual-tower retrieval training.",
                "item_vec = torch.cat([product_emb, category_emb], dim=1)",
            ),
            (
                "LightGCN",
                "Used tensor math, dot product, save/load, and simple tensor indexing for graph embeddings.",
                "score = torch.sum(user_emb * item_emb, dim=1)",
            ),
        ],
    ),
]

for title, items in sections:
    story.append(Paragraph(title, styles["Section"]))
    for name, detail, example in items:
        if "Input:" in detail and "Output:" in detail and "Purpose:" in detail:
            input_text = detail.split("Input: ", 1)[1].split(" Output: ", 1)[0].strip()
            output_text = detail.split(" Output: ", 1)[1].split(" Purpose: ", 1)[0].strip()
            purpose_text = detail.split(" Purpose: ", 1)[1].strip()
        else:
            input_text = "Not explicitly listed"
            output_text = "See the example or notebook context"
            purpose_text = detail.strip()
        story.append(
            Paragraph(
                f"<b>{name}</b><br/><b>Input:</b> {input_text}<br/><b>Output:</b> {output_text}<br/><b>Purpose:</b> {purpose_text}",
                styles["BodySmall"],
            )
        )
        story.append(Preformatted(example, styles["MonoSmall"]))
    story.append(Spacer(1, 0.08 * inch))

story.append(Paragraph("Quick reminders", styles["Section"]))
story.append(
    Paragraph(
        "Use <b>expand</b> for broadcasting without copying, <b>repeat</b> for actual duplication, <b>view</b> when the tensor is contiguous, and <b>reshape</b> when you want the safer option.",
        styles["BodySmall"],
    )
)
story.append(
    Paragraph(
        "Use <b>transpose</b> for swapping two axes and <b>permute</b> for reordering many axes.",
        styles["BodySmall"],
    )
)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(letter[0] - 48, 24, f"Page {doc.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(output_path)
