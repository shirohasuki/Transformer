from datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input

# 1 准备测试数据
enc_inputs, dec_inputs, dec_outputs = make_data()
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
enc_inputs, _, _ = next(iter(loader))

# 2 加载模型
model = torch.load('model.pth')

# 3 推理预测
predict_dec_input = test(model, enc_inputs[0].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
predict, _, _, _ = model(enc_inputs[0].view(1, -1).to(device), predict_dec_input) # 取索引为0的句子测试
predict = predict.data.max(1, keepdim=True)[1] # 临时用max代替了softmax

print([src_idx2word[int(i)] for i in enc_inputs[0]], '->', [tgt_idx2word[n.item()] for n in predict.squeeze()])
