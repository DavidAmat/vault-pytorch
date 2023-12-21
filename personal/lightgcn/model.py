import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch_geometric.utils import structured_negative_sampling
from utils import bpr_loss, get_metrics, ndcg_at_k

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.emb_users = nn.Embedding(num_embeddings=self.num_users, embedding_dim=dim_h, dtype=torch.float32)
        self.emb_items = nn.Embedding(num_embeddings=self.num_items, embedding_dim=dim_h, dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_values = edge_values
        self.adj_mat = self.compute_norm_adj_matrix(edge_index, edge_values)
        self.sp_adj_mat = self._convert_sp_mat_to_csr_tensor(self.adj_mat)
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
        self.sp_val_adj_mat = self._convert_sp_mat_to_csr_tensor(self.val_adj_mat)

        # Move sparse matrices to cuda if device is cuda
        if device == torch.device('cuda'):
            self.sp_val_adj_mat = self.sp_val_adj_mat.cuda()
            self.sp_adj_mat = self.sp_adj_mat.cuda()            
    
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
    def _convert_sp_mat_to_csr_tensor(X):
        coo = X.tocoo().astype(np.float32)  # Convert to float32
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        index = torch.stack([row, col])
        data = torch.tensor(coo.data, dtype=torch.float32)  # Convert to float32
        sparse_tensor = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return sparse_tensor.to_sparse_csr() 

    @staticmethod
    def efficient_adj_matrix(edge_values, edge_indices, num_rows, num_cols):
        # Extract rows (users) and cols (items)
        row, col = edge_indices
        
        # M + N users and items
        num_tot = num_rows + num_cols
        
        # New indexing considering A matrix being a M+N
        values_tot = np.hstack((edge_values, edge_values))

        # Displace the items_ids by M units to form the item_id in the M+N matrix
        row_tot = np.hstack((
            row, col + num_rows
        ))
        col_tot = np.hstack((
            col + num_rows, row
        ))

        A_coo = sp.coo_matrix(
            (values_tot, (row_tot, col_tot)),
            shape=(num_tot, num_tot)
        )
        return A_coo
    
    def compute_norm_adj_matrix(self, edge_index, edge_values, is_valid=False):

        # Efficient Adjacency matrix as a COO matrix
        adj_mat = self.efficient_adj_matrix(
            edge_values=edge_values,
            edge_indices=edge_index,
            num_rows=self.num_users,
            num_cols=self.num_items
        )
        # Row-summation (to get degrees) is better with CSR format
        adj_mat = adj_mat.tocsr()
        
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
        sampled_pos_neg_edges = self.pos_neg_edges[:, index]
        
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
        embf_users, emb0_users, embf_items, emb0_items = self.forward(is_valid=True)

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
    
        # In CPU otherwise OOO error in GPU
        ratings = torch.matmul(
            self.emb_users.weight.to("cpu").to(torch.float32), 
            self.emb_items.weight.T.to("cpu").to(torch.float32)
        )

        # Get ratings by embeddings dot products
        # WARNING: in CUDA this gives OOO error
        # ratings = torch.matmul(self.emb_users.weight, self.emb_items.weight.T)

        # Exclude interactions in the train_set
        excl_user_indices, excl_item_indices = self.edge_index
        ratings[excl_user_indices, excl_item_indices] = -1024

        # get the top k recommended items for each user
        _, top_K_items = torch.topk(ratings, k=topk_recs)

        # -------------- #
        # Get metrics
        # -------------- #

        # Precision and Recall at K        
        l_prec_recall = get_metrics(
            top_rec_items=top_K_items,
            ground_truth=self.edge_index_val,
            k_list=k_list,
            num_users=self.num_users,
            num_items=self.num_items,
        )

        # NDCG at K (list of tuples so that we can add to the l_metrics tuples)
        l_ndcg = [
            (ndcg_at_k(
                top_rec_items=top_K_items, 
                ground_truth=self.edge_index_val, 
                k=kk,
                num_users=self.num_users
            ),) 
            for kk in k_list
        ]

        l_metrics = [(epoch,) + l_prec_recall[idx] + l_ndcg[idx] for idx in range(len(k_list))]
        # Convert to dataframe
        #df_metrics_epoch = pd.DataFrame(l_metrics, columns=["epoch", "K", "TP", "FP", "P", "precision", "recall", "ndcg"])

        return l_metrics