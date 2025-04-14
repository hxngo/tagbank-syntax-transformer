class TAGDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 max_length: int = 512,
                 vocab_file: str = None):
        """
        TAG 데이터셋 초기화
        
        Args:
            data_path: TAG 파일 경로
            max_length: 최대 시퀀스 길이
            vocab_file: 어휘 파일 경로 (선택사항)
        """
        super().__init__()
        
        self.max_length = max_length
        self.data = []
        self.labels = []
        
        # TAG 파일 파싱
        trees = parse_tag_file(data_path)
        
        # 특성 추출
        for tree in trees:
            features = extract_features_from_tag_tree(tree)
            if len(features['tokens']) <= self.max_length:
                self.data.append(features)
                self.labels.append(features['label'])
        
        # 어휘 구축 또는 로드
        if vocab_file and os.path.exists(vocab_file):
            self.vocab = torch.load(vocab_file)
        else:
            self.build_vocab()
            if vocab_file:
                torch.save(self.vocab, vocab_file)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 시퀀스 패딩
        seq_len = len(item['tokens'])
        pad_len = self.max_length - seq_len
        
        # 어텐션 마스크 생성
        attention_mask = torch.ones((self.max_length, 1, 1))
        if pad_len > 0:
            attention_mask[seq_len:] = 0
        
        return {
            'token_ids': self.pad_sequence(item['token_ids'], pad_len),
            'pos_tags': self.pad_sequence(item['pos_tags'], pad_len),
            'elem_types': self.pad_sequence(item['elem_types'], pad_len),
            'syntax_features': self.pad_features(item['syntax_features'], pad_len),
            'tag_relations': self.pad_relations(item['tag_relations'], pad_len),
            'attention_mask': attention_mask,
            'label': torch.tensor(item['label'], dtype=torch.long)
        }
    
    def pad_sequence(self, seq, pad_len):
        if pad_len > 0:
            return torch.cat([
                torch.tensor(seq, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ])
        return torch.tensor(seq[:self.max_length], dtype=torch.long)
    
    def pad_features(self, features, pad_len):
        if pad_len > 0:
            return torch.cat([
                torch.tensor(features, dtype=torch.float),
                torch.zeros((pad_len, features.shape[1]), dtype=torch.float)
            ])
        return torch.tensor(features[:self.max_length], dtype=torch.float)
    
    def pad_relations(self, relations, pad_len):
        seq_len = len(relations)
        if pad_len > 0:
            padded = torch.zeros((self.max_length, self.max_length, relations.shape[2]))
            padded[:seq_len, :seq_len] = torch.tensor(relations)
            return padded
        return torch.tensor(relations[:self.max_length, :self.max_length], dtype=torch.float) 