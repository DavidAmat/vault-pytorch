import numpy as np
import scipy.sparse as sp
import torch


def sparse_matrix_memory_usage(input_matrix: sp):
    # Calculate memory usage in MB
    sparse_matrix = input_matrix.tocsr()
    total_memory_mb = (
        sparse_matrix.data.nbytes +
        sparse_matrix.indices.nbytes +
        sparse_matrix.indptr.nbytes
    ) / (1024 ** 2)  # Convert bytes to megabytes

    print(f"Total memory usage: {total_memory_mb:.5f} MB")
    
def sparse_tensor_memory_usage(sparse_tensor: torch.sparse_coo):
    # Calculate memory usage in bytes
    element_size = sparse_tensor.element_size()
    total_elements = sparse_tensor._nnz()  # _nnz() returns the number of non-zero elements

    total_memory_bytes = element_size * total_elements

    # Convert bytes to megabytes
    total_memory_mb = total_memory_bytes / (1024 ** 2)

    print(f"Total memory usage: {total_memory_mb:.4f} MB")
    

def sum_rows_sparse_tensor(sparse_tensor_coo: torch.sparse_coo):

    # Convert from Pytorch Sparse Tensor to Scipy COO Matrix Sparse
    scipy_sp_coo = sp.coo_matrix(
        (sparse_tensor_coo.coalesce().values().numpy(), 
         sparse_tensor_coo.coalesce().indices().numpy()), 
        shape=sparse_tensor_coo.size()
    )

    # Do the CSR change to then sum row-wise (per user) all the occurences
    scipy_sp_csr = scipy_sp_coo.tocsr()
    return np.array(scipy_sp_csr.sum(axis=1)).flatten()
    
    
def get_metrics(
    top_rec_items: torch.Tensor,
    ground_truth: torch.Tensor,
    num_users: int,
    num_items: int,
    k_list: list
):
    """
    Get the precision and recall for the recommendations (top_rec_items)
    given a ground truth interaction tensor. Let's deep dive on the format of each of them
    
    Args
    ----------
    top_rec_items: torch.tensor
        Tensor of size [users, item_recs] where item_recs is the number of recommended items
        per each user. This matrix usually is the ouput of a torch.topk operation to get
        the top recommended item ids per each user: _, top_rec_items = torch.topk(ratings, k=item_recs)
        Don't confuse this k, the k_list we will apply will take the item_recs and slice it into the top
        k item recommendations per user, this way we will simulate different ranking performances.
        The values of this tensor are item indices, their i,j location mean the following:
            i: index of the user
            j: position in the ranking of that item index on the recommendations of user i-th
    
    ground_truth: torch.tensor
        Tensor of size [2, num_interactions]. 
        Inspired by the edge_index of Pytorch Geometric, this is a 2-D tensor.
        First row [0,:] has user indices
        Second row [1,:] has item indices
        Each column [:, 0] has a user-item interaction. For example, if [:,0] slice is [0,2]
            this means that user_id=0 interacted with item_id=2
        There will be as many columns as interactions
        
    k_list: list
        List of K's we will iterate to compute Prec@K and Rec@K metrics
        
    num_users: int
        Number of users (rows)

    num_items: int
        Number of items (columns)

    Examples:
    # Each row is a user
    # Each column represents the ranking of the item_ids ranked by embedding dot product of user x item embs
    top_rec_items = torch.tensor([
            [2, 0, 1],
            [1, 0, 2],
            [0, 1, 2],
            [0, 1, 2],
            [1, 0, 2]])

    # First row: user ids
    # Second row: item ids
    # Each column is a user-item interaction
    ground_truth = torch.tensor([[0, 1, 2],
                                 [2, 1, 0]])
                                 
    # Call function:
    # get_metrics(top_rec_items, ground_truth, [2])
    [(2, 3.0, 3.0, 3.0, 0.5, 1.0)]
    
    This results indicates that indeed, precision is 50% and recall is 100%
    since user_id=0 interacted with item_id=2 (first column ground_truth)
    and indeed in the ranking of the recs of user_id=0 (first row of top_rec_items: [2, 0, 1]) 
    the item_id that is in the first position of the ranking is 2.
    Since k=2, we are taking 2 ranking positions for that user [2,0] hence, since 0
    is not in the ground truth, this counts as a False Positive
    """
    d_results = []

    # Sparse Tensor of Ground Truth User-Items interactions
    sp_truth = torch.sparse_coo_tensor(
        ground_truth,
        torch.ones_like(ground_truth[0,:], dtype=torch.float32),
        size=(num_users, num_items)
    )

    # Set the value of k
    for k in k_list:

        # ------------------------------ #
        #     Recs interaction matrix
        # ------------------------------ #

        # Slice the recommendations into the k-th item recommended
        top_K_items = top_rec_items[:, :k]

        # Flatten the dense tensor of the item_ids (originally in form n_users x k)
        item_indices = top_K_items.flatten()

        # Generate row and column indices (repeating k times each user to reach length n_users x k)
        user_indices = torch.arange(num_users).repeat(k)

        # Sparse Tensor of K-Recs
        sp_recs_matrix = torch.sparse_coo_tensor(
            torch.stack([user_indices, item_indices]),
            torch.ones_like(item_indices, dtype=torch.float32),
            size=(num_users, num_items)
        )

        # ------------------------------ #
        #     Correct recs
        # ------------------------------ #
        # True Positives (point-wise multiplication of K-Recs * Truth)
        correct_recs = torch.mul(sp_recs_matrix, sp_truth)

        # True Positives and All Positives per user
        true_positives_per_user = sum_rows_sparse_tensor(correct_recs)
        all_positives_per_user = sum_rows_sparse_tensor(sp_truth)
        false_positives_per_user = k - true_positives_per_user

        # Select only users that have at least 1 positive in ground truth
        users_ids_with_positives = np.where(all_positives_per_user >= 1)[0]

        # Take the TP, FP and all P for the users with at least 1 positive
        true_positives = true_positives_per_user[users_ids_with_positives]
        false_positives = false_positives_per_user[users_ids_with_positives]
        all_positives = all_positives_per_user[users_ids_with_positives]

        # Compute precision and recall at k
        tp = true_positives.sum().item()
        fp = false_positives.sum().item()
        all_p = all_positives.sum().item()
        precision = tp / (tp+fp)
        recall = tp / (all_p)
        d_results.append((
            k, 
            tp, 
            fp, 
            all_p,
            precision,
            recall
        ))
        # Convert d_results to a dataframe with this
        # df_results = pd.DataFrame(d_results, columns=["K", "TP", "FP", "P", "precision", "recall"])
    return d_results


def ndcg_at_k(
    top_rec_items: torch.Tensor,
    ground_truth: torch.Tensor,
    k: int,
    num_users: int
):
    """"
    Get the NDCG per user and averages it returning the mean NDCG.
    Let's deep dive on the format of each of the inputs
    
    Args
    ----------
    top_rec_items: torch.tensor
        Tensor of size [users, item_recs] where item_recs is the number of recommended items
        per each user. This matrix usually is the ouput of a torch.topk operation to get
        the top recommended item ids per each user: _, top_rec_items = torch.topk(ratings, k=item_recs)
        Don't confuse this k, the k_list we will apply will take the item_recs and slice it into the top
        k item recommendations per user, this way we will simulate different ranking performances.
        The values of this tensor are item indices, their i,j location mean the following:
            i: index of the user
            j: position in the ranking of that item index on the recommendations of user i-th
    
    ground_truth: torch.tensor
        Tensor of size [2, num_interactions]. 
        Inspired by the edge_index of Pytorch Geometric, this is a 2-D tensor.
        First row [0,:] has user indices
        Second row [1,:] has item indices
        Each column [:, 0] has a user-item interaction. For example, if [:,0] slice is [0,2]
            this means that user_id=0 interacted with item_id=2
        There will be as many columns as interactions
        
    Returns
    ----------
    Average NDCG per user
    """
    assert top_rec_items.shape[1] >= k, "Make sure columns of top_rec_items are >= k"
    ranking_k_recs = top_rec_items[:, :k]
    user_mask_recs = torch.zeros(ranking_k_recs.shape, dtype=bool)
    user_mask_recs_ideal = torch.zeros(ranking_k_recs.shape, dtype=bool)
    
    # Create the weights of the log tensor
    logs_tensor = torch.Tensor(1./np.log2(np.arange(2, k + 2)))

    for user_id in range(num_users):
        # In the ground truth tensor, which is the column ids that this user is present
        idx_interaction_col_userid = ground_truth[0] == user_id

        # Take the item_ids that this user interacted with according to the ground_truth tensor
        # and the index mask we have gathered filtering by user_id
        set_items_interacted_with_user = ground_truth[1][idx_interaction_col_userid]

        # If the user ordered 10 items but k=2, of course we won't be able to retrieve all 10
        # hence, we want to take the min between 10 and 2 (2) so that we assume for this
        # k=2 ranking that the user orered two items
        num_items_iteracted_with_user = min(len(set_items_interacted_with_user), k)

        # Check that from the final ranking of item_ids, if a item_id is in the set of 
        # items interacted by this user
        mask_user = torch.isin(ranking_k_recs[user_id], set_items_interacted_with_user)

        # Create the ideal mask as if the true number of interacted items of that user
        # were placed in the top positions of the recommendations
        mask_user_ideal = torch.zeros_like(mask_user)
        mask_user_ideal[:num_items_iteracted_with_user] = True

        # Update the row of the final user_mask
        user_mask_recs[user_id,:] = mask_user   
        user_mask_recs_ideal[user_id,:] = mask_user_ideal


    # Mask Boolean to integers (0 and 1)
    user_mask_recs = user_mask_recs.float()
    user_mask_recs_ideal = user_mask_recs_ideal.float()

    # Calculate DCG an IDCG per user: here we apply the discounting weight
    # of each position in the ranking
    dcg = torch.mul(user_mask_recs, logs_tensor).sum(dim=1)
    idcg = torch.mul(user_mask_recs_ideal, logs_tensor).sum(dim=1)
    
    # Make sure we don't count in the final mean users that DON'T HAVE ANY
    # item interaction, hence, even though we recommended those users something
    # we don't want to penalize since we don't have any source truth. Hence,
    # we want to exclude those users from the NDCG final mean
    mask_non_null_idcg = idcg > 0

    # Normalized DCG (NDCG) per user
    ndcg = (dcg[mask_non_null_idcg] / idcg[mask_non_null_idcg]).nan_to_num(0)

    # Mean NDCG per user
    mean_ndcg = torch.mean(ndcg)
    return mean_ndcg.item()

def bpr_loss(emb_users_final, emb_users, 
             emb_pos_items_final, emb_pos_items, 
             emb_neg_items_final, emb_neg_items,
             LAMBDA=1e-6
):
    # Tensors size
    #    emb_users_final: [batch_size, embedding_dim]
    
    # Regularization term (norm of the ORIGINAL embedding, not the propagated one)
    reg_loss = LAMBDA * (emb_users.norm().pow(2) +
                        emb_pos_items.norm().pow(2) +
                        emb_neg_items.norm().pow(2))
    
    # We want the dot product to be computed like:
    # emb_users_final[0] @ emb_pos_items_final[0]
    # emb_users_final[1] @ emb_pos_items_final[1]
    # to achieve this, this is the same as doing pairwise multiplication (torch.mul)
    # of the batched tensors and then apply a sumation over the rows
    # so that we will end up with a pos_ratings of size [batch_size]
    # which each index (0, 1) will have the dot product of user embedding vs. item embedding

    # Dot product of the propagated user embedding with the propagated positive item embedding
    pos_ratings = torch.mul(emb_users_final, emb_pos_items_final).sum(dim=-1)
    
    # Dot product of the propagated user embedding with the propagated negative item embedding
    neg_ratings = torch.mul(emb_users_final, emb_neg_items_final).sum(dim=-1)

    
    # careful using this, because bpr_loss here is always positive in softplus
    # hence, we always want the loss function to contribute positively
    # bpr_loss = torch.mean(torch.nn.functional.softplus(pos_ratings - neg_ratings))
    
    # if we choose the paper implementation, we use LogSigmoid (always negative)
    # this is why we will change the sign from negative to positive in the final loss computation
    bpr_loss = torch.mean(torch.nn.functional.logsigmoid(pos_ratings - neg_ratings))
    
    # consider adding regularization loss (always positive since we are adding norms)
    return -bpr_loss + reg_loss