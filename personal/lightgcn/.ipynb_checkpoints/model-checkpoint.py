

class LightGCN(nn.Module):
    def __init__(self, 
                 num_users, num_items, 
                 edge_index, edge_values,
                 edge_index_val=None, edge_values_val=None,
                 num_layers=4, dim_h=64, batch_size=2):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.emb_users = nn.Embedding(num_embeddings=self.num_users, embedding_dim=dim_h)
        self.emb_items = nn.Embedding(num_embeddings=self.num_items, embedding_dim=dim_h)
        self.edge_index = edge_index
        self.edge_values = edge_values
        self.adj_mat = self.compute_norm_adj_matrix(edge_index, edge_values)
        self.sp_adj_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat)
        self.alpha = 1/(self.num_layers+1)
        self.batch_size = batch_size

        # self.convs = nn.ModuleList(LGConv() for _ in range(num_layers))

        nn.init.normal_(self.emb_users.weight, std=0.01)
        nn.init.normal_(self.emb_items.weight, std=0.01)
        
        # Construct positive and negative edges
        self._generate_positives_negative_edges()
        
        # Validation adjacency matrix
        self.edge_index_val = edge_index_val
        self.edge_values_val = edge_values_val
        self.val_adj_mat = self.compute_norm_adj_matrix(edge_index_val, edge_values_val, is_valid=True)        
        self.sp_val_adj_mat = self._convert_sp_mat_to_sp_tensor(self.val_adj_mat)
    
    def _generate_positives_negative_edges(self):
        edge_index = self.edge_index

        # Generate negative sample indices
        # IMPORTANT! let's consider only as num_nodes the size of the items set
        # this is to avoid the case where num_users > num_items and we sample
        # from a user_id that is higher than a item_id and we will be providing 
        # and index for the item_id that does not exist
        edge_index = structured_negative_sampling(edge_index, num_nodes=self.num_items)
        
        # edge_index: Tuple of 3 tensors
        # tensor1: indices of user node
        # tensor2: indices of item node (positive interaction with user)
        # tensor3: indices of the item node (negative interaction with user)
        self.pos_neg_edges = torch.stack(edge_index, dim=0)       
        
        
    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def compute_norm_adj_matrix(self, edge_index, edge_values, is_valid=False):
        num_users = self.num_users
        num_items = self.num_items
        # Interaction matrix
        R = sp.coo_matrix((
            edge_values, 
            (edge_index[0], edge_index[1])),
            shape=(num_users, num_items))
        R = R.tolil()

        # Save interaction matrix
        if not is_valid:
            self.R = R
        else:
            self.R_valid = R
        
        # Adjacency matrix
        MN = self.num_users + self.num_items
        adj_mat = sp.dok_matrix((MN, MN), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        
        # Fill adjacency matrix
        adj_mat[:num_users, num_users:] = R
        adj_mat[num_users:, :num_users] = R.T
        
        # Degrees
        rowsum = np.array(adj_mat.sum(1))
        
        # Inverse of the Degree matrix
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        # Normalized Adjacency Matrix
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        return norm_adj
    
    def sample_mini_batch(self):
        # Generate BATCH_SIZE random indices (size: [batch_size])
        index = np.random.choice(range(self.edge_index.shape[1]), size=self.batch_size)
        
        # With such index, select the positive-negative edge pairs generated before
        # first dimension is user, second is positive item, third is negative item
        # sampled_pos_neg_edges: [3 x len(indices)]
        sampled_pos_neg_edges = model.pos_neg_edges[:, index]
        
        # user_indices: [[1 x len(indices)]] (first row of pos_neg_edges sampled)
        # etc...
        user_indices, pos_item_indices, neg_item_indices = sampled_pos_neg_edges.numpy()

        return user_indices, pos_item_indices, neg_item_indices
        

    def forward(self, is_valid=False):
        # Keep track of starting embeddings for feeding into the BPR Loss 
        # for regularizing the learned embedding params
        emb0_users = self.emb_users.weight
        emb0_items = self.emb_items.weight
                
        # Embedding is dimension M + N
        emb = torch.cat([emb0_users, emb0_items])
        embs = [emb]

        # For each layer
        for layer_i in range(self.num_layers):
            if not is_valid:
                emb = torch.sparse.mm(self.sp_adj_mat, emb)
            else:
                emb = torch.sparse.mm(self.sp_val_adj_mat, emb)
            embs.append(emb)

        emb_final = self.alpha * torch.mean(torch.stack(embs, dim=1), dim=1)

        embf_users, embf_items = torch.split(emb_final, [self.num_users, self.num_items])

        return embf_users, emb0_users, embf_items, emb0_items
    
    # EVALUATION
    
    # Validation loss
    def valid_loss(self):
        # Forward pass using the validation adjacency matrix
        emb_users_final, emb_users, emb_items_final, emb_items = self.forward(is_valid=True)

        # Choose negative sampling
        user_indices, pos_item_indices, neg_item_indices = structured_negative_sampling(
            self.edge_index_val, 
            num_nodes=self.num_items,
            contains_neg_self_loops=False
        )

        # Applying sample indices
        s_embf_users, s_emb0_users = embf_users[user_indices], emb0_users[user_indices]
        s_embf_items_pos, s_emb0_items_pos = embf_items[pos_item_indices], emb0_items[pos_item_indices]
        s_embf_items_neg, s_emb0_items_neg = embf_items[neg_item_indices], emb0_items[neg_item_indices]


        # Loss computation
        valid_loss = bpr_loss(
            s_embf_users, s_emb0_users, 
            s_embf_items_pos, s_emb0_items_pos, 
            s_embf_items_neg, s_emb0_items_neg
        ).item()

        #recall, ndcg = get_metrics(model, edge_index, exclude_edge_indices)

        return valid_loss # , recall, ndcg
    
    def get_val_metrics(self, epoch: int, topk_recs=10, k_list=[1,2,3]):
    
        # Get ratings by embeddings dot products
        ratings = torch.matmul(self.emb_users.weight, self.emb_items.weight.T)


        # Exclude interactions in the train_set
        excl_user_indices, excl_item_indices = self.edge_index
        ratings[excl_user_indices, excl_item_indices] = -1024

        # get the top k recommended items for each user
        _, top_K_items = torch.topk(ratings, k=topk_recs)

        # Get metrics
        model.edge_index_val
        l_metrics = get_metrics(
            top_rec_items=top_K_items,
            ground_truth=self.edge_index_val,
            k_list=k_list
        )
        l_metrics = [(epoch,) + tup for tup in l_metrics]
        # Convert to dataframe
        #df_metrics_epoch = pd.DataFrame(l_metrics, columns=["epoch", "K", "TP", "FP", "P", "precision", "recall"])

        return l_metrics