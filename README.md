import itertools, functools, math
from typing import Iterator, Tuple, Iterable
import matplotlib.pyplot as plt

Polygon = Tuple[Tuple[float, float], ...]

# ----------------------------
# 1) Генераторы бесконечных последовательностей
# ----------------------------
def count_2D(start=(0,0), step=(1,1)) -> Iterator[Tuple[float,float]]:
    x0, y0 = start
    dx, dy = step
    n = 0
    while True:
        yield (x0 + n*dx, y0 + n*dy)
        n += 1


def gen_rectangle(size=(1,1), start=(0,0), step=(2,0)) -> Iterator[Polygon]:
    for x, y in count_2D(start, step):
        w, h = size
        yield ((x,y), (x+w,y), (x+w,y+h), (x,y+h))


def gen_triangle(size=1, start=(0,0), step=(2,0)) -> Iterator[Polygon]:
    for x, y in count_2D(start, step):
        s = size
        yield ((x,y), (x+s,y), (x+s/2, y + s*math.sqrt(3)/2))


def gen_hexagon(size=1, start=(0,0), step=(3,0)) -> Iterator[Polygon]:
    for x, y in count_2D(start, step):
        s = size
        pts = []
        for k in range(6):
            ang = math.pi/3 * k
            pts.append((x + s*math.cos(ang), y + s*math.sin(ang)))
        yield tuple(pts)

# ----------------------------
# 2) Преобразования
# ----------------------------
def tr_translate(vec):
    dx, dy = vec
    def _trans(polygon: Polygon) -> Polygon:
        return tuple((x+dx, y+dy) for x, y in polygon)
    return _trans


def tr_rotate(angle, center=(0,0)):
    cx, cy = center
    def _rot(polygon: Polygon) -> Polygon:
        ca, sa = math.cos(angle), math.sin(angle)
        res = []
        for x, y in polygon:
            x0, y0 = x-cx, y-cy
            xr = x0*ca - y0*sa + cx
            yr = x0*sa + y0*ca + cy
            res.append((xr, yr))
        return tuple(res)
    return _rot


def tr_symmetry(axis='x'):
    def _sym(polygon: Polygon) -> Polygon:
        if axis not in ('x','y'): raise ValueError
        return tuple((x, -y) if axis=='x' else (-x, y) for x,y in polygon)
    return _sym


def tr_homothety(k, center=(0,0)):
    cx, cy = center
    def _hom(polygon: Polygon) -> Polygon:
        return tuple(((x-cx)*k+cx, (y-cy)*k+cy) for x,y in polygon)
    return _hom

# ----------------------------
# 3) Фильтры
# ----------------------------
def area(polygon: Polygon) -> float:
    pts = polygon
    s = 0
    for (x1,y1), (x2,y2) in zip(pts, pts[1:]+pts[:1]):
        s += x1*y2 - x2*y1
    return abs(s)/2


def side_lengths(polygon: Polygon) -> Tuple[float,...]:
    pts = polygon
    res = []
    for (x1,y1), (x2,y2) in zip(pts, pts[1:]+pts[:1]):
        res.append(math.hypot(x2-x1, y2-y1))
    return tuple(res)


def is_convex(polygon: Polygon) -> bool:
    pts = polygon
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    signs = []
    for i in range(len(pts)):
        o, a, b = pts[i], pts[(i+1)%len(pts)], pts[(i+2)%len(pts)]
        signs.append(cross(o,a,b))
    return all(s>=0 for s in signs) or all(s<=0 for s in signs)


def flt_convex_polygon(poly_iter: Iterable[Polygon]) -> Iterator[Polygon]:
    return filter(is_convex, poly_iter)


def flt_angle_point(point):
    def _flt(poly_iter):
        return filter(lambda p: point in p, poly_iter)
    return _flt


def flt_square(min_area):
    def _flt(poly_iter):
        return filter(lambda p: area(p) < min_area, poly_iter)
    return _flt


def flt_short_side(min_len):
    def _flt(poly_iter):
        return filter(lambda p: min(side_lengths(p)) < min_len, poly_iter)
    return _flt


def point_in_poly(pt, polygon: Polygon) -> bool:
    x,y = pt
    cnt = False
    pts = polygon
    for (x1,y1),(x2,y2) in zip(pts, pts[1:]+pts[:1]):
        if ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1)+x1): cnt = not cnt
    return cnt


def flt_point_inside(point):
    def _flt(poly_iter):
        return filter(lambda p: is_convex(p) and point_in_poly(point,p), poly_iter)
    return _flt


def flt_polygon_angles_inside(poly0: Polygon):
    angles = set(poly0)
    def _flt(poly_iter):
        return filter(lambda p: is_convex(p) and any(v in angles for v in p), poly_iter)
    return _flt

# ----------------------------
# 4) Декораторы
# ----------------------------
def decorator_from_filter(filt):
    def deco(fn):
        def wrapper(*args, **kwargs):
            new_args = [filt(arg) if isinstance(arg, Iterable) else arg for arg in args]
            return fn(*new_args, **kwargs)
        return wrapper
    return deco


def decorator_from_transform(transf):
    def deco(fn):
        def wrapper(*args, **kwargs):
            new_args = [map(transf, arg) if isinstance(arg, Iterable) else arg for arg in args]
            return fn(*new_args, **kwargs)
        return wrapper
    return deco

# примеры декораторов:
@decorator_from_filter(flt_convex_polygon)
def dec_convex(polys): return list(polys)
@decorator_from_filter(flt_angle_point((0,0)))
def dec_angle(polys): return list(polys)
@decorator_from_filter(flt_square(2))
def dec_square(polys): return list(polys)
@decorator_from_filter(flt_short_side(1))
def dec_short(polys): return list(polys)
@decorator_from_filter(flt_point_inside((0.5,0.5)))
def dec_inside(polys): return list(polys)
base_poly = ((0,0),(1,0),(1,1),(0,1))
@decorator_from_filter(flt_polygon_angles_inside(base_poly))
def dec_poly_angles(polys): return list(polys)
@decorator_from_transform(tr_translate((1,1)))
def dec_trans(polys): return list(polys)
@decorator_from_transform(tr_rotate(math.pi/4))
def dec_rot(polys): return list(polys)
@decorator_from_transform(tr_symmetry('y'))
def dec_sym(polys): return list(polys)
@decorator_from_transform(tr_homothety(2))
def dec_hom(polys): return list(polys)

# ----------------------------
# 5) Агрегаторы
# ----------------------------
def agr_origin_nearest(polys):
    def red(best, poly):
        for v in poly:
            if best is None or (v[0]**2+v[1]**2)<(best[0]**2+best[1]**2): best=v
        return best
    return functools.reduce(red, polys, None)

def agr_max_side(polys): return functools.reduce(lambda m,p: max(m, max(side_lengths(p))), polys, 0)

def agr_min_area(polys): return functools.reduce(lambda m,p: area(p) if m is None or area(p)<m else m, polys, None)

def agr_perimeter(polys): return functools.reduce(lambda s,p: s+sum(side_lengths(p)), polys, 0)

def agr_area(polys): return functools.reduce(lambda s,p: s+area(p), polys, 0)

# ----------------------------
# 6) Склейка
# ----------------------------
def zip_polygons(*iters):
    for group in zip(*iters): yield tuple(itertools.chain.from_iterable(group))

def zip_tuple(*iters):
    for group in zip(*iters): yield tuple(sum(group, ()))

# ----------------------------
# 7) Тесты и визуализации
# ----------------------------
if __name__ == "__main__":
    # 2) семь фигур
    seq = itertools.islice(itertools.chain(gen_rectangle((1,2)), gen_triangle(1), gen_hexagon(1)), 7)
    plt.figure(figsize=(4,4))
    for poly in seq:
        xs, ys = zip(*(poly+(poly[0],)))
        plt.plot(xs, ys)
    plt.axis('equal'); plt.title("7 фигур")
    plt.show()

    # 3a) три параллельные ленты под 30°
    plt.figure(figsize=(5,5))
    for ofs in [(-1,0),(0,0),(1,0)]:
        tape = itertools.islice(map(tr_translate(ofs), map(tr_rotate(math.pi/6), gen_rectangle((1,0.5), step=(2,0)))),5)
        for p in tape:
            xs, ys = zip(*(p+(p[0],))); plt.plot(xs, ys)
    plt.axis('equal'); plt.title("3 ленты 30°"); plt.show()

    # 3b) две пересекающиеся ленты
    plt.figure(figsize=(5,5))
    t1 = itertools.islice(map(tr_translate((2,1)), map(tr_rotate(math.pi/5), gen_triangle(0.8, step=(1.5,0)))),6)
    t2 = itertools.islice(map(tr_translate((0,2)), map(tr_rotate(-math.pi/4), gen_rectangle((1,1), step=(1.5,0)))),6)
    for p in t1: xs,ys=zip(*(p+(p[0],))); plt.plot(xs,ys)
    for p in t2: xs,ys=zip(*(p+(p[0],))); plt.plot(xs,ys)
    plt.axis('equal'); plt.title("2 пересекающиеся"); plt.show()

    # 3c) две симметричные ленты треугольников
    plt.figure(figsize=(5,5))
    t1 = itertools.islice(map(tr_rotate(math.pi/6), gen_triangle(1, step=(2,0))),6)
    t2 = itertools.islice(map(tr_rotate(-math.pi/6), gen_triangle(1, start=(0,2), step=(2,0))),6)
    for p in t1: xs,ys=zip(*(p+(p[0],))); plt.plot(xs,ys)
    for p in t2: xs,ys=zip(*(p+(p[0],))); plt.plot(xs,ys)
    plt.axis('equal'); plt.title("2 симметричные"); plt.show()

    # 3d) квадраты разного масштаба между y=x/2 и y=-x/3
    plt.figure(figsize=(5,5))
    xs = [-5,5]
    plt.plot(xs, [x/2 for x in xs]); plt.plot(xs, [-x/3 for x in xs])
        # 3d) квадраты разного масштаба между y=x/2 и y=-x/3
    plt.figure(figsize=(5,5))
    xs = [-5,5]
    plt.plot(xs, [x/2 for x in xs]); plt.plot(xs, [-x/3 for x in xs])
    for k in range(1,6):
        # передаём полигон как единый кортеж
        p = tr_homothety(k)(((0,0),(1,0),(1,1),(0,1)))
        xs2, ys2 = zip(*(p+(p[0],)))
        plt.plot(xs2, ys2)
    plt.axis('equal'); plt.title("квадраты между линиями"); plt.show()

    # 4.1) 6 фигур с короткой стороной <1.5
    base = (tr_homothety(1+0.3*i)(((0,0),(1,0),(1,1),(0,1))) for i in range(10))
    filtered6 = list(itertools.islice(flt_short_side(1.5)(base),6))
    print("6 фигур <1.5:", filtered6)

    # 4.2) ≤4 фигур из ≥15 с min side <0.8
    many = (tr_homothety(0.5+0.1*i)(((0,0),(1,0),(1,1),(0,1))) for i in range(15))
    small = list(itertools.islice(flt_short_side(0.8)(many),4))
    print("<=4 фигур <0.8:", small)

    # 4.3) фильтрация пересечений
    quads = [((i*0.5,i*0.5),(i*0.5+1,i*0.5),(i*0.5+1,i*0.5+1),(i*0.5,i*0.5+1)) for i in range(15)]
    def seg_int(a,b,c,d):
        def ori(p,q,r): return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
        return ori(a,b,c)*ori(a,b,d)<0 and ori(c,d,a)*ori(c,d,b)<0
    def poly_int(p,q):
        return any(seg_int(a,b,c,d) for a,b in zip(p,p[1:]+p[:1]) for c,d in zip(q,q[1:]+q[:1]))
    nonint=[]
    for p in quads:
        if not any(poly_int(p,q) for q in nonint): nonint.append(p)
    print("непересекающихся осталось:", len(nonint))

    # 5) декораторы
    sample = list(itertools.islice(gen_triangle(1, step=(2,0)),5))
    print("dec_convex:", dec_convex(sample))
    print("dec_angle:", dec_angle(sample))
    print("dec_square:", dec_square(sample))
    print("dec_short:", dec_short(sample))
    print("dec_inside:", dec_inside(sample))
    print("dec_poly_angles:", dec_poly_angles(sample))
    print("dec_trans:", dec_trans(sample)[:3])
    print("dec_rot:", dec_rot(sample)[:3])
    print("dec_sym:", dec_sym(sample)[:3])
    print("dec_hom:", dec_hom(sample)[:3])

    # 5) агрегаторы
    print("Nearest:", agr_origin_nearest(sample))
    print("Max side:", agr_max_side(sample))
    print("Min area:", agr_min_area(sample))
    print("Perim:", agr_perimeter(sample))
    print("Area:", agr_area(sample))

    # 6) zip
    a = list(itertools.islice(gen_triangle(1, step=(3,0)),3))
    b = list(itertools.islice(gen_rectangle((1,1), start=(0,-3), step=(3,0)),3))
    print("zip_polygons:", list(zip_polygons(a,b)))
    print("zip_tuple:", list(zip_tuple([(1,1),(2,2),(3,3),(4,4)], [(2,2),(3,3),(4,4),(5,5)])))
