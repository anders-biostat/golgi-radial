library( tidyverse )
library( reticulate )
library( EBImage )
library( magick )

mrc <- import("mrc")

load_dv_file <- function( filename ) {
   img <- mrc$imread( filename )
   img <- array( aperm(img,c(2,1,3,4)), c( dim(img)[1], dim(img)[2], dim(img)[3], dim(img)[4] ) )
   img <- aperm( img, c(2,1,3,4) )
   dimnames(img) <- list( image=NULL, channel=c("DAPI", "centr", "Golgi"), x=NULL, y=NULL ) 
   img
}

my_display <- function( img, segm=NULL, file=NULL ) {
   img <- aperm( img, c(2,3,1) )
   img <- img[,,c(3,2,1)]
   for( j in 1:3 )
      img[,,j] <- img[,,j] / max(img[,,j]) 
   if( !is.null(segm) ) {
      
      #anno <- ( dilate(segm>0) - (segm>0) )/2
      storage.mode(segm) <- "integer"
      segm_ids <- setdiff( unique(as.integer(segm)), 0 )
      anno <- matrix( 0, nrow=nrow(segm), ncol=ncol(segm) )
      for( i in segm_ids ) {
         anno <- anno + (segm==i) - erode(segm==i, makeBrush(3,"diamond")) }
      
      for( cell in segm_ids ) {
         mean_row <- sum( (segm==cell) * row(anno) ) / sum(segm==cell)
         mean_col <- sum( (segm==cell) * col(anno) ) / sum(segm==cell)
         sprintf("label:%d",cell) %>% image_read() %>% image_convert(type="grayscale") %>% 
            image_data() %>% as.integer() %>% drop() %>% t() -> m
         start_row <- min( max( 1, round( mean_row-nrow(m)/2 ) ), nrow(anno)-nrow(m) )
         start_col <- min( max( 1, round( mean_col-ncol(m)/2 ) ), ncol(anno)-ncol(m) )
         anno[ start_row:(start_row+nrow(m)-1), start_col:(start_col+ncol(m)-1) ] <- 
            1 - ( 1 - anno[ start_row:(start_row+nrow(m)-1), start_col:(start_col+ncol(m)-1) ] ) * m/255 
      }
      img[,,1] <- 1 - (1-img[,,1]) * (1-anno)
      img[,,2] <- 1 - (1-img[,,2]) * (1-anno)
   }
   img <- Image( img, colormode="color" )
   if( is.null(file) ) {
      display( img )
   } else {
      writeImage( img, file )
   }
}

margin <- array( 0, dim=dim(imgs)[3:4] )
margin[1:3,] <- 1
margin[,1:3] <- 1
margin[nrow(margin)+(-2:0),] <- 1
margin[,ncol(margin)+(-2:0)] <- 1


for( fn in list.files( "imgs_zmax", "dv$" ) ) {
   print(fn)
   imgs <- load_dv_file( str_c( "imgs_zmax/", fn ) )
   
   for( frame in 1:dim(imgs)[1] ) {
      print(frame)
      img <- imgs[frame,,,]
      for( ch in 1:3) 
         img[ch,,] <- img[ch,,] / max(img[ch,,])
      
      mask_dapi <- img["DAPI",,] > otsu(img["DAPI",,]) * 1.2
      img_golgi <- img["Golgi",,]
      img_golgi <- filter2( img_golgi, makeBrush(7,"Gaussian"))
      mask_golgi <- img_golgi > otsu(img_golgi) * .7
      mask <- mask_dapi | mask_golgi
      mask <- erode( mask, makeBrush( 11, "box" ) )
      segm <- bwlabel( mask )
      for( i in 1:max(segm)) {
         if( sum(segm==i) < 500 ) {
            mask[segm==i] <- 0 } }
      segm <- bwlabel(mask)
      segmprop <- propagate(mask,segm)
      
      for( i in 1:max(segm) ) {
         if ( sum( (segm==i) * margin ) > 0 )
            segmprop[segmprop==i] <- 0
      }
      
      my_display( img, segmprop,
         sprintf( "tmp/%s_%2d.png", str_remove( fn, "\\.dv"), frame ) )
   }
}







for( fn in list.files( "imgs_zmax", "dv$" ) ) {
   print(fn)
   imgs <- load_dv_file( str_c( "imgs_zmax/", fn ) )
   
   for( frame in 1:dim(imgs)[1] ) {
      print(frame)
      img <- imgs[frame,,,]
      for( ch in 1:3) 
         img[ch,,] <- img[ch,,] / max(img[ch,,])
      
      mask_dapi <- img["DAPI",,] > otsu(img["DAPI",,])*1.2
      mask_dapi <- dilate( mask_dapi, makeBrush( 3, "box" ) )
      dapi_segm <- bwlabel(mask_dapi)
      for( i in 1:max(dapi_segm)) {
         if( sum(dapi_segm==i) < 500 ) {
            mask_dapi[dapi_segm==i] <- 0 } }
      mask_dapi <- dilate( mask_dapi, makeBrush( 3, "box" ) )
      my_display( img, bwlabel(mask_dapi) )#, #segm>0  
      
      img_golgi_gamma <- img["Golgi",,]
      img_golgi_gamma <- filter2( img_golgi_gamma, makeBrush(7,"Gaussian"))
      mask_golgi <- img_golgi_gamma > otsu(img_golgi_gamma)*.9
      mask_golgi <- erode( mask_golgi, makeBrush( 9, "box" ) )
      mask <- mask_dapi | mask_golgi
      mask <- erode( mask, makeBrush( 11, "box" ) )
      segm <- bwlabel( mask )
      
      my_display( img, mask_dapi )#, #segm>0  
         sprintf( "tmp/%s_%2d.png", str_remove( fn, "\\.dv"), frame ) )
   }
}
############################

















image( img["DAPI",,], asp=1 )
image( img["Golgi",,], asp=1 )




get_voronoi_areas <- function(img) {

   dapi_img <- img[1,,]
   golgi_img <- img[3,,]
   dapi_img <- dapi_img / max(dapi_img)
   golgi_img <- golgi_img / max(golgi_img)

   dapi_segments <- ( dapi_img > otsu(dapi_img) )
   segments <- dapi_segments + ( golgi_img > otsu(golgi_img) )
   segments[segments>1] <- 1
   segments <- erode( segments, makeBrush( 9, "disc" ) )
   segments <- dilate( segments, makeBrush( 9, "disc" ) )
   
   labelled_segments <- bwlabel(segments)
   segments_with_DAPI <- sapply( 1:max(labelled_segments), function(i)
      sum( (labelled_segments==i) * dapi_segments ) > 1000 )
   
   border <- array( 0, dim=dim(dapi_img) )
   border[1:3,] <- 1
   border[,1:3] <- 1
   border[nrow(border)+(-2:0),] <- 1
   border[,ncol(border)+(-2:0)] <- 1
   
   segments_at_border <- sapply( 1:max(labelled_segments), function(i)
      sum( border * segments * (labelled_segments==i) ) > 0 )

   list( 
      segments = segments, 
      voronoi = propagate( segments, bwlabel( segments ) ),
      info = data.frame(
         label = 1:max(labelled_segments),
         with_DAPI = segments_with_DAPI,
         at_border = segments_at_border )
   )
}

annotate_image <- function( img, v ) {
   
   borders <- ( v$voronoi - filter2( v$voronoi, makeBrush(3,"box"))/9 )^2 > 1e-4
   
   textlayer <- image_blank(512, 512, color="black")
   for( i in 1:max(v$voronoi) ) {
      cr <- sum( (v$voronoi==i) * row(v$voronoi) ) / sum(v$voronoi==i)
      cc <- sum( (v$voronoi==i) * col(v$voronoi) ) / sum(v$voronoi==i)
      mark <- {if ( v$info$with_DAPI[i] & !v$info$at_border[i] ) "✓" else "✗"}
      textlayer <- image_annotate( textlayer, sprintf("%s%s",i,mark), size=20, color="white", 
                                   location=sprintf("+%f+%f", cr, cc) )
   }
   borders <- borders + t(as.numeric(image_data(textlayer))[,,1])
   
   dimg <- EBImage::Image( aperm( img, c(2,3,1) )[,,c(3,2,1)], colormode="color" )
   for( j in 1:3 )
      dimg[,,j] <- dimg[,,j] / max(dimg[,,j]) 
   dimg[,,1] <- dimg[,,1] + borders/3
   dimg[,,2] <- dimg[,,2] + borders/3
   
   dimg
}


get_golgi_variance_for_cell <- function( img, cell_voronoi ) {

   dapi_img <- img[1,,]
   golgi_img <- img[3,,]
   dapi_img <- dapi_img / max(dapi_img)
   golgi_img <- golgi_img / max(golgi_img)

   golgi_img_wobg <- golgi_img - median(golgi_img)
   golgi_img_wobg[golgi_img_wobg<0] <- 0
   
   mean_col <- sum( golgi_img_wobg * cell_voronoi * col(golgi_img) ) / 
      sum( golgi_img_wobg * cell_voronoi )
   mean_row <- sum( golgi_img_wobg * cell_voronoi * row(golgi_img) ) / 
      sum( golgi_img_wobg * cell_voronoi )

   distsq_to_cm <- ( row(golgi_img) - mean_row )^2 + ( col(golgi_img) - mean_col )^2
   variance <- sum( golgi_img_wobg * cell_voronoi * distsq_to_cm ) / 
      sum( golgi_img_wobg * cell_voronoi )
   
   variance   
}

process_image_set <- function(imgs, prefix) {
   map( 1:dim(imgs)[1], function(img_idx) {
      v <- get_voronoi_areas( imgs[img_idx,,,] )
      
      writeImage(
         annotate_image( imgs[img_idx,,,], v ),
         sprintf( "anno/%s__%02d.png", prefix, img_idx ) )
      
      v$info %>%
      filter( with_DAPI & !at_border ) %>%
      rowwise() %>%
      mutate( variance = get_golgi_variance_for_cell( 
         imgs[img_idx,,,], v$voronoi==label ) )
   }) %>%
   bind_rows( .id="img_idx" )
}


list.files("imgs_zmax") %>%
set_names() %>%
map( function(filename) {
   cat( filename, "\n" )
   imgs <- load_dv_file( str_c( "imgs_zmax/", filename ) )
   process_image_set( imgs, str_remove( filename, ".dv" ) )
}) %>%
bind_rows( .id="img_file" ) -> result

tibble( img_file = list.files("imgs_zmax") ) %>%
mutate( genotype = rep( c( "KO", "WT" ), each=4 ) ) %>%
mutate( treatment = c( "2h WO", "2h drug", "no drug", "30min WO", 
      "2h drug", "2h WO", "30min WO", "no drug" ) ) -> sample_table

result %>%
left_join( sample_table ) %>%
unite( sample, c( "genotype", "treatment" ), sep=" / " ) %>%
ggplot() + 
   ggbeeswarm::geom_beeswarm( aes( x=sample, y=variance ) ) +
   scale_y_log10() +
   theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
# 

######################


result %>%
left_join( sample_table ) %>%
lm( log(variance) ~ genotype * treatment, . ) %>% anova()
   


img <- imgs[img_idx,,,]
v <- get_voronoi_areas(img)
image(v$segments)






dapi_img <- img[1,,]
golgi_img <- img[3,,]
dapi_img <- dapi_img / max(dapi_img)
golgi_img <- golgi_img / max(golgi_img)

dapi_segments <- dapi_img > otsu(dapi_img)

segments <- dapi_segments + ( golgi_img > otsu(golgi_img) )
segments[segments>1] <- 1
segments <- dilate( segments, makeBrush( 29, "disc" ) )

labelled_segments <- bwlabel(segments)
valid_segments <- sapply( 1:max(labelled_segments), function(i)
   sum( (labelled_segments==i) * dapi_segments ) > 4000 )



list( 
   segments = segments, 
   voronoi = propagate( segments, bwlabel( segments ) ) )


img <- imgs[2,,,]
v <- get_voronoi_areas( img )

borders <- ( v$voronoi - filter2( v$voronoi, makeBrush(3,"box"))/9 )^2 > 1e-4

textlayer <- image_blank(512, 512, color="black")
for( i in 1:max(v$voronoi) ) {
   cr <- sum( (v$voronoi==i) * row(v$voronoi) ) / sum(v$voronoi==i)
   cc <- sum( (v$voronoi==i) * col(v$voronoi) ) / sum(v$voronoi==i)
   textlayer <- image_annotate( textlayer, as.character(i), size=20, color="white", 
      location=sprintf("+%f+%f", cr, cc) )
}
borders <- borders + t(as.numeric(image_data(textlayer))[,,1])

dimg <- EBImage::Image( aperm( img, c(2,3,1) )[,,c(3,2,1)], colormode="color" )
for( j in 1:3 )
   dimg[,,j] <- dimg[,,j] / max(dimg[,,j]) 
dimg[,,1] <- dimg[,,1] + borders/3
dimg[,,2] <- dimg[,,2] + borders/3
dimg
display( EBImage::Image( dimg, colormode="color" ) )
