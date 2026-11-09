#include "scene/point_to_plane_icp.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include "math/robust_loss.h"

namespace nox::native::scene {
namespace {

bool solve6(double matrix[6][7],double x[6]) {
    for(int column=0;column<6;++column){
        int pivot=column;
        for(int row=column+1;row<6;++row){
            if(std::abs(matrix[row][column])>
               std::abs(matrix[pivot][column]))pivot=row;
        }
        if(std::abs(matrix[pivot][column])<1e-10)return false;
        if(pivot!=column){
            for(int k=column;k<7;++k)std::swap(matrix[pivot][k],matrix[column][k]);
        }
        for(int row=column+1;row<6;++row){
            const double factor=matrix[row][column]/matrix[column][column];
            for(int k=column;k<7;++k)matrix[row][k]-=factor*matrix[column][k];
        }
    }
    for(int row=5;row>=0;--row){
        double value=matrix[row][6];
        for(int k=row+1;k<6;++k)value-=matrix[row][k]*x[k];
        x[row]=value/matrix[row][row];
    }
    return true;
}

}

IcpResult PointToPlaneIcp::align(
    const std::vector<IcpCorrespondence>& correspondences,
    const math::SE3d& initial,
    const IcpOptions& options) const {
    IcpResult result;
    result.pose=initial;

    if(correspondences.size()<6)return result;

    auto evaluate=[&](const math::SE3d& pose,double& cost,std::size_t& inliers){
        double squared=0.0;
        double totalWeight=0.0;
        cost=0.0;
        inliers=0;
        for(const auto& correspondence:correspondences){
            const auto transformed=pose.transform(correspondence.source);
            const double dx=transformed.x-correspondence.target.x;
            const double dy=transformed.y-correspondence.target.y;
            const double dz=transformed.z-correspondence.target.z;
            const double distance=std::sqrt(dx*dx+dy*dy+dz*dz);
            if(distance>options.maxCorrespondenceDistance)continue;
            const double residual=
                correspondence.normal.x*dx+
                correspondence.normal.y*dy+
                correspondence.normal.z*dz;
            const double baseWeight=std::max(0.0,correspondence.weight);
            const double robustWeight=baseWeight*
                math::RobustLoss::weight(
                    math::RobustLossKind::Huber,
                    residual,
                    options.robustScale);
            if(robustWeight<=0.0)continue;
            squared+=robustWeight*residual*residual;
            totalWeight+=robustWeight;
            ++inliers;
            cost+=baseWeight*math::RobustLoss::cost(
                math::RobustLossKind::Huber,
                residual,
                options.robustScale);
        }
        if(totalWeight<=0.0)return std::numeric_limits<double>::infinity();
        return std::sqrt(squared/totalWeight);
    };

    double initialCost=0.0;
    std::size_t initialInliers=0;
    result.initialRms=evaluate(result.pose,initialCost,initialInliers);
    if(!std::isfinite(result.initialRms))return result;

    double currentCost=initialCost;
    for(std::uint32_t iteration=0;iteration<options.iterations;++iteration){
        double h[6][6]{};
        double b[6]{};
        for(const auto& correspondence:correspondences){
            const auto transformed=result.pose.transform(correspondence.source);
            const double dx=transformed.x-correspondence.target.x;
            const double dy=transformed.y-correspondence.target.y;
            const double dz=transformed.z-correspondence.target.z;
            const double distance=std::sqrt(dx*dx+dy*dy+dz*dz);
            if(distance>options.maxCorrespondenceDistance)continue;

            const double residual=
                correspondence.normal.x*dx+
                correspondence.normal.y*dy+
                correspondence.normal.z*dz;
            const double weight=std::max(0.0,correspondence.weight)*
                math::RobustLoss::weight(
                    math::RobustLossKind::Huber,
                    residual,
                    options.robustScale);
            if(weight<=0.0)continue;

            const auto& rawNormal=correspondence.normal;
            const double normalLength=std::sqrt(
                rawNormal.x*rawNormal.x+
                rawNormal.y*rawNormal.y+
                rawNormal.z*rawNormal.z);
            if(normalLength<1e-9)continue;
            const math::Vec3d n=rawNormal*(1.0/normalLength);
            const math::Vec3d cross{
                transformed.y*n.z-transformed.z*n.y,
                transformed.z*n.x-transformed.x*n.z,
                transformed.x*n.y-transformed.y*n.x,
            };
            const double J[6]={
                n.x,n.y,n.z,
                cross.x,cross.y,cross.z,
            };
            for(int row=0;row<6;++row){
                b[row]-=weight*J[row]*residual;
                for(int col=0;col<6;++col){
                    h[row][col]+=weight*J[row]*J[col];
                }
            }
        }

        for(int i=0;i<6;++i)h[i][i]+=options.damping;
        double minDiagonal=std::numeric_limits<double>::infinity();
        double maxDiagonal=0.0;
        for(int i=0;i<6;++i){
            minDiagonal=std::min(minDiagonal,h[i][i]);
            maxDiagonal=std::max(maxDiagonal,h[i][i]);
        }
        result.conditionEstimate=
            maxDiagonal/std::max(minDiagonal,1e-12);
        if(!std::isfinite(result.conditionEstimate)||
           result.conditionEstimate>1e10)break;

        double augmented[6][7]{};
        for(int row=0;row<6;++row){
            for(int col=0;col<6;++col)augmented[row][col]=h[row][col];
            augmented[row][6]=b[row];
        }

        double delta[6]{};
        if(!solve6(augmented,delta))break;

        const math::Twist6d twist{
            {delta[3],delta[4],delta[5]},
            {delta[0],delta[1],delta[2]},
        };
        const auto candidate=math::SE3d::exp(twist).compose(result.pose);

        double candidateCost=0.0;
        std::size_t candidateInliers=0;
        const double candidateRms=evaluate(
            candidate,candidateCost,candidateInliers);
        if(!std::isfinite(candidateRms)||candidateCost>currentCost+1e-9){
            break;
        }

        result.pose=candidate;
        currentCost=candidateCost;
        result.finalRms=candidateRms;
        result.inliers=candidateInliers;
        result.iterations=iteration+1;

        const double stepNorm=std::sqrt(
            delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]+
            delta[3]*delta[3]+delta[4]*delta[4]+delta[5]*delta[5]);
        if(stepNorm<options.convergence){
            result.converged=true;
            break;
        }
    }

    if(result.iterations==0){
        result.finalRms=result.initialRms;
        result.inliers=initialInliers;
    }

    result.confidence=std::clamp(
        std::exp(-result.finalRms/
                 std::max(options.robustScale,1e-6))*
        (static_cast<double>(result.inliers)/
         static_cast<double>(correspondences.size())),
        0.0,1.0);
    return result;
}
}
