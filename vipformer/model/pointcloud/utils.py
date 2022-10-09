import torch
import torch.nn as nn
import torch.nn.functional as F


def divide_patches(points, num_groups, group_size):
    '''
        Args:
            points: [batch_size, num_points, point_channels], here `point_channels` can be > 3
        Return:
            centers: [batch_size, num_groups, point_channels]
            neighbors: [batch_size, num_groups, group_size, point_channels]
    '''
    batch_size, num_points, point_channels = points.shape
    # centers: [batch_size, num_groups, point_channels]
    centers = fps(points, num_groups)
    # knn to get the neighborhood of each center
    # idx: [batch_size, num_groups, group_size]
    neighbor_idx = knn_point(group_size, points[:, :, :3], centers[:, :, :3])
    
    # neighbor_idx_base: [batch_size, 1, 1]
    neighbor_idx_base = torch.arange(0, batch_size, device=points.device).view(-1, 1, 1) * num_points
    # neighbor_idx: [batch_size, num_groups, group_size]
    neighbor_idx = neighbor_idx + neighbor_idx_base
    # neighbor_idx: [batch_size * num_groups * group_size]
    neighbor_idx = neighbor_idx.view(-1)

    # points.reshape(batch_size * num_points, -1) -> [batch_size * num_points, point_channels]
    # neighbors: [batch_size * num_groups * group_size, point_channels]
    neighbors = points.reshape(batch_size * num_points, -1)[neighbor_idx, :]
    # neighbors: [batch_size, num_groups, group_size, point_channels]
    neighbors = neighbors.reshape(batch_size, num_groups, group_size, point_channels)
    # normalize only point coordinates, which are first 3 dimensions
    #   centers.unsqueeze(2): [batch_size, num_groups, 1, point_channels]
    #   neighbors: [batch_size, num_groups, group_size, point_channels]
    neighbors[:, :, :3] = neighbors[:, :, :3] - centers.unsqueeze(2)[:, :, :3]

    return neighbors, centers


def fps(pts, number):
    '''
        Args:
            pts: [batch_size, num_points, point_channels]
            number: the number of points FPS will return
        Return:
            fps_point: [batch_size, num_groups, point_channels]
    '''
    
    fps_idx = farthest_point_sample(pts, number)
    fps_pts = index_points(pts, fps_idx)

    return fps_pts


def farthest_point_sample(pts, npoint):
    """
        Args:
            pts: pointcloud data, [B, N, C]. Here `C` is not necessary 3, it can be 6.
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
    """
    device = pts.device
    B, N, _ = pts.shape
    # centroids: [B, npoint]
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    # distance: [B, N]
    distance = torch.ones(B, N, device=device) * 1e10   # don't overlook `* 1e10`
    # farthest: [B]
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    # batch_indices: [B]
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        # centroid: [B, 1, 3]
        centroid = pts[batch_indices, farthest, :3].view(B, 1, 3)
        # dist: [B, N]
        dist = torch.sum((pts[:,:,:3] - centroid) ** 2, -1)
        # distance: [B, N]
        distance = torch.min(distance, dist)
        # farthest: [B]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points, idx):
    """
        Args:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def knn_point(nsample, xyz, new_xyz):
    '''
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    '''
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)

    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class Group2Emb(nn.Module):
    '''
        project each group to corresponding embedding
        reference: point-bert
            ------- light-weight PointNet in the paper
    '''
    def __init__(self, dim_model, point_channels=3):
        super().__init__()
        self.dim_model = dim_model
        self.point_channels = point_channels
        self.first_conv = nn.Sequential(
            nn.Conv1d(point_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1)
        )
        self.second_conv = nn.Sequential(
            # in_channels = 2 * 128
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.dim_model, 1)
        )

    def forward(self, point_groups):
        '''
            Args:
                point_groups : [batch, num_groups, group_size, point_channels]
            Return:
                groups_emb : [batch, num_groups, dim_model]
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.point_channels)
        # feature: [batch*num_groups, 128, group_size]
        feature = self.first_conv(point_groups.transpose(2,1))
        # feature_global: [batch*num_groups, 128, 1]
        # ------ MaxPool in each group
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        # feature: [batch*num_groups, 256, group_size]
        # ------ Concatenate feature_global and feature
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)
        # feature: [batch*num_groups, dim_model, group_size]
        feature = self.second_conv(feature)
        # feature: [batch*num_groups, dim_model]
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.dim_model)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class Sequential(nn.Sequential):
    def forward(self, *x):
        for module in self:
            if type(x) == tuple:
                x = module(*x)
            else:
                x = module(x)
        return x
