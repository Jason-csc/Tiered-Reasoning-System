import torch.nn as nn
import torch
import transformers

class TieredModel(nn.Module):
    def __init__(self, MODEL_PATH, lstm_hiddenDim=100, bilstm=False):
        super(TieredModel, self).__init__()
        self.roberta = transformers.AutoModel.from_pretrained(MODEL_PATH)
        for param in self.roberta.parameters():
            param.requires_grad = False
        input_dim = 1024 if "large" in MODEL_PATH else 768
        print("\nLM parameters:",input_dim)
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hiddenDim, batch_first=True, bidirectional=bilstm)
        self.drop_out = nn.Dropout(p=0.2)
        # self.conflictClassify1 = nn.Linear(2*5*lstm_hiddenDim,10) if bilstm else nn.Linear(5*lstm_hiddenDim,10)
        self.conflictClassify1 = nn.Linear(2*5*lstm_hiddenDim,lstm_hiddenDim) if bilstm else nn.Linear(5*lstm_hiddenDim,lstm_hiddenDim)
        self.conflictClassify2 = nn.Linear(lstm_hiddenDim,10)
        self.Activation = nn.ReLU()
        self.plauClassify = nn.Linear(2*10,2)
        torch.nn.init.normal_(self.conflictClassify1.weight, std=0.02)
        torch.nn.init.normal_(self.conflictClassify2.weight, std=0.02)
        torch.nn.init.normal_(self.plauClassify.weight, std=0.02)
        
        
    def forward(self, inputData): #inputdata = dataset['story_input']
        conflicts = []
        conflicts_actv = []
        conflicts_task1 = []
        for story in inputData:
            sentences = []
            for inputs in story:
                out = self.roberta(input_ids=inputs['input_ids'],attention_mask=inputs['mask_ids'],output_hidden_states=True)
                out = out.hidden_states[-1]
                out = torch.mean(out, dim=1).squeeze()
                out = self.drop_out(out)
                sentences.append(out)
            assert len(sentences) == 5
            setences = (torch.stack(sentences)).view(1,5,-1)
            lstm_out, _ = self.bilstm(setences)
            lstm_out = self.drop_out(lstm_out.view(1,-1))
            conflict_out = self.conflictClassify2(self.Activation(self.conflictClassify1(lstm_out)))
            # conflict_out = self.conflictClassify1(lstm_out)
            conflicts.append(conflict_out)
            conflicts_actv.append(conflict_out.view(-1))
        plausible_out = self.Activation(torch.cat(conflicts_actv))
        plausible_out = self.plauClassify(plausible_out)
        if plausible_out[0] > plausible_out[1]:
            conflicts = conflicts[0]
        else:
            conflicts = conflicts[1]
        return plausible_out.view(1,-1), conflicts
        # plausible_out = self.Activation(torch.cat(conflicts_task1))
        # plausible_out = self.plauClassify2(plausible_out)
        # return plausible_out.view(1,-1), None

       
