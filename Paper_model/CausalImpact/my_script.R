install.packages("CausalImpact")
library(CausalImpact)

# 創建CausalImpact對象
impact <- CausalImpact(data, pre.period, post.period)

# 繪製結果
plot(impact)
