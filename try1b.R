library( tiff )
library( EBImage )
library( tidyverse )
library( ggbeeswarm )
library( magick )
library( mirai )

# Check image

imgs <- readTIFF( "imgs_segm/Segmented images_CEP250KO/250KO_nodrug/F7_NODRUG_CEP250KO.tif", all = TRUE )
str(imgs)
image( imgs[[1]][,,1], asp=1 )  # Golgi
image( imgs[[1]][,,2], asp=1 )  # Centr
image( imgs[[1]][,,3], asp=1 )  # DAPI
image( imgs[[2]][,,1], asp=1 )  # segm  (sometimes in imgs[[3]])
image( imgs[[2]][,,2], asp=1 )  # same
image( imgs[[2]][,,3], asp=1 )  # same
image( imgs[[2]][,,4], asp=1 )  # nothing

get_distsq_to_center <- function(img) {
   mean_col <- sum(  img * col(img) ) / sum(img)
   mean_row <- sum(  img * row(img) ) / sum(img)
   ( row(img) - mean_row )^2 + ( col(img) - mean_col )^2  
}

get_radial_variance <- function( img ) {
   distsq_to_cm <- get_distsq_to_center( img )
   sum( img * distsq_to_cm ) / sum(img)
} 

get_radial_ecdf <- function( img ) {
   distsq_to_cm <- get_distsq_to_center( img )
   a <- sapply( 1:max(nrow(img),ncol(img)), function(r)
      sum( img * (distsq_to_cm < r^2) ) )
   a / max(a)
}  


process_tiff_file <- function( filename ) {
   print(filename)
   imgs <- readTIFF( str_c( "imgs_segm/", filename ), all = TRUE )
   
   img_golgi <- imgs[[1]][,,1]
   img_centr <- imgs[[1]][,,2]
   img_dapi  <- imgs[[1]][,,3]
   img_segm  <- imgs[[length(imgs)]][,,1]
  
   segm_labeled <- bwlabel( img_segm<.5 )
   
   statistic <- lapply( 1:max(segm_labeled), function(cell)
       get_radial_ecdf( img_golgi * (segm_labeled==cell) ) )
   
   tibble(
      filename = filename,
      cell_idx = 1:length(statistic),
      statistic = statistic )
}


list.files( "imgs_segm/", pattern="tif$", recursive=TRUE ) %>%
set_names() %>%
map( process_tiff_file ) %>%
list_rbind() %>%
mutate( basename = filename %>% str_split_i("/",-1) %>% str_remove(".tif") ) %>%
separate( basename, c( "frame", "cond", "genotype" ), "_+" ) -> res

ggplot(res) +
   geom_quasirandom( aes( x=cond, y=sqrt(variance), col=cond ) )

res %>% unnest() %>% unite( "cell", cond, frame, cell_idx ) %>% group_by(cell) %>% mutate(r=row_number()) %>% ggplot() + geom_line(aes(x=r,y=statistic,group=cell)) + xlim(0,100)

# annotate

for( filename in list.files( "imgs_segm/", pattern="tif$", recursive=TRUE ) ) {
   
   imgs <- readTIFF( str_c( "imgs_segm/", filename ), all = TRUE )
   img_anno <- imgs[[1]]
   img_segm <- imgs[[length(imgs)]][,,1]

   anno <- ( (img_segm<.1) - erode(img_segm<.1) )/2
   segm_labeled <- bwlabel(img_segm<.1)
   for( cell in 1:max(segm_labeled) ) {
      mean_row <- sum( (segm_labeled==cell) * row(anno) ) / sum(segm_labeled==cell)
      mean_col <- sum( (segm_labeled==cell) * col(anno) ) / sum(segm_labeled==cell)
      sprintf("label:%d",cell) %>% image_read() %>% image_convert(type="grayscale") %>% 
         image_data() %>% as.integer() %>% drop() %>% t() -> m
      start_row <- round( mean_row-nrow(m)/2 )
      start_col <- round( mean_col-ncol(m)/2 )
      anno[ start_row:(start_row+nrow(m)-1), start_col:(start_col+ncol(m)-1) ] <- 
      1 - ( 1 - anno[ start_row:(start_row+nrow(m)-1), start_col:(start_col+ncol(m)-1) ] ) * m/255 
   }
   
   img_anno[,,1] <- 1 - (1-img_anno[,,1]) * (1-anno)
   img_anno[,,2] <- 1 - (1-img_anno[,,2]) * (1-anno)

   writeImage( 
      Image(img_anno,colormode="color"), 
      str_c( "imgs_anno/", filename %>% str_split_i("/",-1) %>% str_remove(".tif"), ".anno.png" ) )
}
  