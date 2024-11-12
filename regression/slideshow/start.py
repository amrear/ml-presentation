from manim import *
from manim_slides import Slide, ThreeDSlide

import numpy as np
import sklearn.linear_model


class Intro(Slide):
    def intro(self):
        title = Text("Introduction to Machine Learning")
        self.play(Write(title))
        self.next_slide()
        self.play(title.animate.scale(0.5).shift(UP * 3.35))
        self.next_slide()
        self.play(FadeOut(title))


class LinearRegression(ThreeDSlide):
    def construct(self):
        title = Text("Linear Regression").scale(2)
        description = Text("A Method to Estimate Numerical Data")
        description.next_to(title, DOWN).scale(0.75)
        self.play(Write(title))
        self.play(Write(description))

        self.next_slide()

        self.play(title.animate.scale(0.5).shift(
            UP * 3.35), FadeOut(description))

        plot = VGroup()
        ax = Axes(
            x_range=[0, 500, 100],
            y_range=[0, 500000, 100000],
            tips=False,
            axis_config={"include_numbers": True},
            x_length=7,
            y_length=5,
        ).add_coordinates()

        plot.add(ax)

        ax.shift(DOWN * 0.5)

        labels = ax.get_axis_labels(
            x_label=r"Size (m^{2})", y_label=r"Price (\$)")

        plot.add(labels)

        self.play(FadeIn(ax))
        self.play(FadeIn(labels))

        size = [193.75, 93.75, 175.0, 50.0, 343.75, 437.5, 412.5, 293.75,
                143.75, 481.25, 312.5, 75.0, 368.75, 475.5, 450.0, 85.3]
        price = [326250.0, 93000.0, 210000.0, 37500.0, 450000.0, 412500.0, 439500.0, 247500.0,
                 300000.0, 435000.0, 495000.0, 90000.0, 270000.0, 105000.0, 450000.0, 150000.0]

        dots = [Dot(ax.coords_to_point(*c), color=GREEN)
                for c in zip(size, price)]
        plot.add(*dots)

        self.play(*[FadeIn(d) for d in dots])

        model = sklearn.linear_model.LinearRegression()

        size = np.reshape(size, (-1, 1))
        price = np.reshape(price, (-1, 1))
        model.fit(size, price)

        graph = ax.plot(lambda x: model.coef_[
                        0][0] * x + model.intercept_[0], x_range=[0, 500], use_smoothing=False, color=RED)
        plot.add(graph)
        self.next_slide()
        self.play(Create(graph))

        self.next_slide()
        self.play(plot.animate.scale(0.8).shift(UP * 0.65))

        equation = Tex(
            fr"$Price = {model.coef_[0][0]:,.1f} \cdot Size + {model.intercept_[0]:,.1f}$")
        equation.shift(DOWN * 3.25).scale(0.7)
        self.play(Write(equation))

        x = 324.1
        y = model.predict(np.reshape([x], (-1, 1)))[0][0]

        self.next_slide()
        equation2 = Tex(fr"$Price({x} m^{2}) = \${y:,.1f}$")

        self.play(equation.animate.shift(UP * 0.4))

        equation2.next_to(equation, DOWN).scale(0.7)

        self.play(Write(equation2))

        prediction = ax.get_lines_to_point(ax.c2p(x, y))
        plot.add(prediction)
        self.play(FadeIn(prediction))

        self.next_slide()
        self.play(FadeOut(prediction), FadeOut(equation2))
        graph2 = ax.plot(lambda x: (model.coef_[0][0] + 100) * x + model.intercept_[
                         0] - 25000, x_range=[0, 500], use_smoothing=False, color=ORANGE)
        equation3 = Tex(
            fr"$Price = {model.coef_[0][0] + 100:,.1f} \cdot Size + {model.intercept_[0] -25000:,.1f}$")
        equation3.next_to(equation, DOWN).scale(0.7)
        self.play(Create(graph2))
        self.play(Write(equation3))
        title2 = Text("Which Model Is Better?").shift(UP * 3.35)
        self.play(ReplacementTransform(title, title2))

        self.next_slide()

        self.play(graph.animate.shift([0.0, 0.0, 0.0]).set_opacity(0.3), graph2.animate.shift([0.0, 0.0, 0.0]).set_opacity(0.3),
                  FadeOut(equation), FadeOut(equation3))

        mse = Tex(
            r"$\displaystyle \operatorname {Error} =\sum _{i=1}^{n}Y_{i}-{\hat {Y_{i}}}$")
        mse.shift(DOWN * 3.25).scale(0.7)
        self.play(Write(mse))

        self.next_slide()

        mse2 = Tex(
            r"$\displaystyle \operatorname {Error} =\sum _{i=1}^{n}\left(Y_{i}-{\hat {Y_{i}}}\right)^{2}$")
        mse2.shift(DOWN * 3.25)
        self.play(ReplacementTransform(mse, mse2))

        self.next_slide()

        mse3 = Tex(
            r"$\displaystyle \operatorname {Error} ={\frac {1}{n}}\sum _{i=1}^{n}\left(Y_{i}-{\hat {Y_{i}}}\right)^{2}$")
        mse3.shift(DOWN * 3.25)
        self.play(ReplacementTransform(mse2, mse3))

        self.next_slide()
        self.play(FadeOut(mse3))
        self.play(FadeIn(equation), graph.animate.shift(
            [0.0, 0.0, 0.0]).set_opacity(1))

        self.next_slide()

        all_preds = model.predict(size).reshape(1, -1)[0]
        size = size.reshape(1, -1)[0]
        pred_dots = [Dot(ax.coords_to_point(*c), color=GREEN)
                     for c in zip(size, all_preds)]
        plot.add(*pred_dots)
        self.play(*[FadeIn(d) for d in pred_dots])

        error_lines = [Line(*pair, color=GREEN)
                       for pair in zip(dots, pred_dots)]
        self.play(*[Create(el) for el in error_lines])
        error1 = Tex(
            f"Error = {np.sum((price.reshape(1, -1)[0] - all_preds) ** 2) / len(all_preds):,.1f}").set_opacity(0)
        error1 = error1.scale(0.7).next_to(equation, RIGHT)
        group1 = VGroup(equation, error1)
        self.play(group1.animate.move_to([0, group1.get_center()[1], 0]))
        self.play(Write(error1.set_opacity(1)))

        self.next_slide()

        self.play(graph.animate.shift([0.0, 0.0, 0.0]).set_opacity(0.3),
                  *[FadeOut(ap) for ap in pred_dots], *[FadeOut(el) for el in error_lines])
        self.play(FadeIn(equation3), graph2.animate.shift(
            [0.0, 0.0, 0.0]).set_opacity(1))

        model2 = sklearn.linear_model.LinearRegression()
        model2.coef_ = np.array([model.coef_[0][0] + 100])
        model2.intercept_ = np.array([model.intercept_[0] - 25000])

        size = size.reshape(-1, 1)
        all_preds2 = model2.predict(size).reshape(1, -1)[0]
        size = size.reshape(1, -1)[0]
        pred_dots2 = [Dot(ax.coords_to_point(*c), color=GREEN)
                      for c in zip(size, all_preds2)]
        plot.add(*pred_dots2)
        self.play(*[FadeIn(d) for d in pred_dots2])

        error_lines2 = [Line(*pair, color=GREEN)
                        for pair in zip(dots, pred_dots2)]
        self.play(*[Create(el) for el in error_lines2])
        error2 = Tex(
            f"Error = {np.sum((price.reshape(1, -1)[0] - all_preds2) ** 2) / len(all_preds2):,.1f}").set_opacity(0)
        error2 = error2.scale(0.7).next_to(equation3, RIGHT)
        group2 = VGroup(equation3, error2)
        self.play(group2.animate.move_to([0, group2.get_center()[1], 0]))
        self.play(Write(error2.set_opacity(1)))

        self.next_slide()

        better_eq = Rectangle(width=group1.width + 0.3,
                              height=group1.height + 0.3)
        better_eq.move_to(group1.get_center())
        self.play(Create(better_eq))

        self.next_slide()

        linear10 = sklearn.linear_model.LinearRegression()
        linear10.coef_ = np.array([0])
        linear10.intercept_ = np.array([0])

        self.play(FadeOut(better_eq), FadeOut(group1), FadeOut(group2), FadeOut(graph), FadeOut(
            graph2), *[FadeOut(el) for el in error_lines2], *[FadeOut(d) for d in pred_dots2])
        title3 = Text("How “Fitting” Takes Place?").shift(UP * 3.35)
        self.play(ReplacementTransform(title2, title3))
        w = ValueTracker(0)
        b = ValueTracker(0)
        equation10 = Tex(
            fr"$Price = {w.get_value():,.1f} \cdot Size + {b.get_value():,.1f}$")
        equation10.shift(DOWN * 3.25).scale(0.7)

        size = size.reshape(-1, 1)
        hello_world = linear10.predict(size)
        error10 = Tex(
            f"Error = {np.sum((price.reshape(1, -1)[0] - hello_world) ** 2) / len(hello_world):,.1f}")
        error10 = error10.scale(0.7).next_to(equation10, RIGHT)

        group10 = VGroup(equation10, error10)
        group10.move_to([0, group10.get_center()[1], 0])

        graph4 = ax.plot(lambda x: w.get_value() * x + b.get_value(),
                         x_range=[0, 500], use_smoothing=False, color=RED)
        self.play(Create(graph4))
        self.play(Write(group10))

        self.next_slide()

        def funcy(m):
            return m.become(
                ax.plot(lambda x: w.get_value() * x + b.get_value(),
                        x_range=[0, 500], use_smoothing=False, color=RED)
            )

        graph4.add_updater(
            funcy
        )

        weights = np.sqrt(np.linspace(0, model.coef_[0][0] ** 2, 10))
        biases = np.sqrt(np.linspace(0, model.intercept_[0] ** 2, 10))

        for we, ba in zip(weights, biases):
            linear10.coef_ = np.array([we])
            linear10.intercept_ = np.array([ba])
            hello_world = linear10.predict(size)

            tmp = Tex(fr"$Price = {we:,.1f} \cdot Size + {ba:,.1f}$")
            tmp.shift(DOWN * 3.25).scale(0.7)
            tmp200 = Tex(
                f"Error = {np.sum((price.reshape(1, -1)[0] - hello_world) ** 2) / len(hello_world):,.1f}")
            tmp200 = tmp200.scale(0.7).next_to(tmp, RIGHT)

            tmpgroup = VGroup(tmp200, tmp)
            tmpgroup.move_to([0, group10.get_center()[1], 0])

            rt = ReplacementTransform(equation10, tmp)
            rt2 = ReplacementTransform(error10, tmp200)
            self.play(
                ApplyMethod(w.set_value, we),
                ApplyMethod(b.set_value, ba),
                rt,
                rt2)
            equation10 = tmp
            error10 = tmp200

        self.next_slide()
        graph4.remove_updater(funcy)

        self.play(FadeOut(ax), FadeOut(group10), FadeOut(graph4),
                  *[FadeOut(d) for d in dots], FadeOut(labels))
        title4 = Text("Overfitting And Underfitting").shift(UP * 3.35)
        self.play(ReplacementTransform(title3, title4))

        plot2 = VGroup()
        ax2 = Axes(
            x_range=[0, np.pi + 0.1, 1],
            y_range=[0, 1.1, 0.2],
            tips=True,
            axis_config={"include_numbers": True},
            x_length=7,
            y_length=5,
        ).add_coordinates()

        plot2.add(ax2)

        ax2.shift(DOWN * 0.5)

        labels2 = ax2.get_axis_labels(x_label=r"x", y_label=r"f(x)")

        plot2.add(labels2)

        self.play(FadeIn(ax2))
        self.play(FadeIn(labels2))

        xx = np.array([0., 0.16534698, 0.33069396, 0.49604095, 0.66138793,
                       0.82673491, 0.99208189, 1.15742887, 1.32277585, 1.48812284,
                       1.65346982, 1.8188168, 1.98416378, 2.14951076, 2.31485774,
                       2.48020473, 2.64555171, 2.81089869, 2.97624567, 3.14159265])

        sinx = np.array([0.01313289,  0.9531357,  0.27303831,  0.56426336,  0.69446266,
                         0.79658221,  0.72262249,  1.0565153,  0.90835232,  1.01992807,
                         1.11962864,  0.85117504,  1.01662771,  0.91446878,  0.67722208,
                         0.6222458,  0.511015,  0.39366855,  0.18442382, 0.03879388])

        dots50 = [Dot(ax2.coords_to_point(*c), color=GREEN)
                  for c in zip(xx.tolist(), sinx.tolist())]
        plot2.add(*dots50)
        self.play(*[FadeIn(d) for d in dots50])

        model50 = sklearn.linear_model.LinearRegression()

        model50.fit(xx.reshape((-1, 1)), sinx.reshape((-1, 1)))

        graph50 = ax2.plot(lambda x: model50.coef_[
            0][0] * x + model50.intercept_[0], x_range=[0, np.pi], use_smoothing=False, color=RED)
        plot2.add(graph50)
        self.next_slide()
        self.play(Create(graph50))

        self.next_slide()

        plr = np.poly1d(np.polyfit(xx, sinx, 50))

        graph60 = ax2.plot(plr, x_range=[0, np.pi, 0.165346982],
                           use_smoothing=False, color=ORANGE)
        plot2.add(graph60)
        self.play(Create(graph60))

        self.next_slide()

        self.play(plot2.animate.scale(0.8).shift(UP * 0.65))

        error65346 = Tex(
            fr"$Error = {abs(sinx - model50.predict(xx.reshape(-1, 1)).reshape(1, -1)).sum():,.1f}$", color=RED)
        error65346.shift(DOWN * 2.65).scale(0.7)

        error5646 = Tex(
            fr"$Error = {abs(sinx - np.array([plr(i) for i in xx])).sum():,.1f}$", color=ORANGE)
        error5646.shift(DOWN * 3.15).scale(0.7)

        self.play(LaggedStart(Write(error65346),
                  Write(error5646), lag_ratio=0.25))

        self.next_slide()

        x = np.array([0., 0.16534698, 0.33069396, 0.49604095, 0.66138793,
                      0.82673491, 0.99208189, 1.15742887, 1.32277585, 1.48812284,
                      1.65346982, 1.8188168, 1.98416378, 2.14951076, 2.31485774,
                      2.48020473, 2.64555171, 2.81089869, 2.97624567, 3.14159265])
        y = np.array([0.13878516,  0.28160998,  0.25253683,  0.30999755,  0.56426281,
                      0.5745891,  0.67355963,  0.78081114,  1.07239615,  1.14142923,
                      0.89806727,  0.92881436,  0.78349175,  0.68117417,  0.76652376,
                      0.80442836,  0.29534375,  0.41772831,  0.24808496, 0.12275737])
        dots60 = [Dot(ax2.coords_to_point(*c), color=BLUE)
                  for c in zip(x, y)]
        plot2.add(*dots60)
        self.play(*[FadeIn(d) for d in dots60])

        title5 = Text("Train And Test Datasets").shift(UP * 3.35)
        self.play(ReplacementTransform(title4, title5))

        self.next_slide()

        error36456 = Tex(
            fr"$Error_{{train}} = {abs(sinx - model50.predict(xx.reshape(-1, 1)).reshape(1, -1)).sum():,.1f}$", color=RED)
        error36456.shift(DOWN * 2.65).scale(0.7)

        error2356 = Tex(
            fr"$Error_{{train}} = {abs(sinx - np.array([plr(i) for i in xx])).sum():,.1f}$", color=ORANGE)
        error2356.shift(DOWN * 3.15).scale(0.7)

        rt3536 = ReplacementTransform(error65346, error36456)
        rt8624 = ReplacementTransform(error5646, error2356)

        self.play(LaggedStart(
                  rt3536, rt8624, lag_ratio=0.25))

        self.next_slide()

        error3546 = Tex(
            fr"$Error_{{test}} = {abs(y - model50.predict(x.reshape(-1, 1)).reshape(1, -1)).sum():,.1f}$", color=RED)
        error3546 = error3546.set_opacity(0).scale(
            0.7).next_to(error36456, RIGHT)

        errorgroup1 = VGroup(error36456, error3546)

        error3646 = Tex(
            fr"$Error_{{test}} = {abs(y - np.array([plr(i) for i in x])).sum():,.1f}$", color=ORANGE)
        error3646 = error3646.set_opacity(
            0).scale(0.7).next_to(error2356, RIGHT)

        errorgroup2 = VGroup(error2356, error3646)

        self.play(LaggedStart(
            errorgroup1.animate.move_to([0, errorgroup1.get_center()[1], 0]),
            errorgroup2.animate.move_to([0, errorgroup2.get_center()[1], 0]), lag_ratio=0.25))

        self.play(Write(error3546.set_opacity(1)),
                  Write(error3646.set_opacity(1)))

        self.next_slide()

        self.play(graph50.animate.set_opacity(0.3), graph60.animate.set_stroke(ORANGE, opacity=0.3))

        plr2 = np.poly1d(np.polyfit(xx, sinx, 2))

        graph70 = ax2.plot(plr2, x_range=[0, np.pi],
                           use_smoothing=False, color=PURPLE)

        plot2.add(graph70)
        self.play(Create(graph70))

        error53351 = Tex(
            fr"$Error_{{train}} = {abs(sinx - np.array([plr2(i) for i in xx])).sum():,.1f}$", color=PURPLE)
        error53351 = error53351.shift(DOWN * 3.65).scale(0.7)

        error6646 = Tex(
            fr"$Error_{{test}} = {abs(y - np.array([plr2(i) for i in x])).sum():,.1f}$", color=PURPLE)
        error6646 = error6646.scale(0.7).next_to(error53351, RIGHT)
        errorgroup3 = VGroup(error53351, error6646)
        errorgroup3.move_to([0, errorgroup3.get_center()[1], 0])
        self.play(Write(errorgroup3))

        self.next_slide()

        self.play(FadeOut(errorgroup1), FadeOut(
            errorgroup2), FadeOut(errorgroup3))
        self.play(plot2.animate.scale(1.25).shift(DOWN * 0.65))
        title6 = Text("What About The Outliers?").shift(UP * 3.35)
        self.play(FadeOut(graph50), FadeOut(graph60), FadeOut(graph70))
        self.play(ReplacementTransform(title5, title6))
        circ = Circle(radius=0.3, color=WHITE).move_to(
            ax2.coords_to_point(0.16534698, 0.9531357))
        self.play(Create(circ))
        plot2.add(circ)

        graph50.set_opacity(0)
        graph60.set_opacity(0)
        graph70.set_opacity(0)

        self.next_slide()

        self.play(FadeOut(plot2))        

        # 3d stuff

        title6454 = Text("We Need To Add More Dimensions").scale(
            2).scale(0.5).shift(UP * 3.35)
        self.play(ReplacementTransform(title6, title6454))
        self.add_fixed_in_frame_mobjects(title6454)

        plot563 = VGroup()
        axes5646 = ThreeDAxes(
            x_range=[0, 500, 100],
            y_range=[0, 5, 1],
            z_range=[0, 500000, 100000]
        )
        x_label43 = axes5646.get_x_axis_label(Text(r"Size"))
        y_label43 = axes5646.get_y_axis_label(Text(r"Ghosts")).shift(UP * 1.8)
        z_label43 = axes5646.get_z_axis_label(Text(r"Price"), buff=1)

        plot563.add(axes5646, x_label43, y_label43, z_label43)

        self.set_camera_orientation(zoom=0.5)
        plot563.move_to(IN * 0.0001)

        self.play(FadeIn(axes5646))
        self.play(Write(x_label43), Write(y_label43))

        self.next_slide()

        size = [193.75, 93.75, 175.0, 50.0, 343.75, 437.5, 412.5, 293.75,
                143.75, 481.25, 312.5, 75.0, 368.75, 475.5, 450.0, 85.3]
        ghosts = [1., 1., 1., 1., 1., 1., 1.,
                  1., 0., 1., 1., 1., 1., 5., 1., 1.]
        price = [326250.0, 93000.0, 210000.0, 37500.0, 450000.0, 412500.0, 439500.0, 247500.0,
                 300000.0, 435000.0, 495000.0, 90000.0, 270000.0, 105000.0, 450000.0, 150000.0]
        
        dots645 = VGroup(*[Dot3D(point=axes5646.coords_to_point(*c), color=GREEN) for c in zip(size, ghosts, price)])

        plot563.add(dots645)

        self.next_slide()

        self.play(FadeIn(dots645))

        self.next_slide()
        self.move_camera(phi=80 * DEGREES, theta=80 * DEGREES)
        self.play(Write(z_label43))

        model = sklearn.linear_model.LinearRegression()
        model.fit(np.array([size, ghosts]).T, price)

        self.next_slide()

        surface = Surface(
            lambda u, v: axes5646.c2p(u, v, model.predict(np.array([[u, v]]))[0]),
            u_range=[0, 500],
            v_range=[0, 5],
            fill_opacity=0.4,
            checkerboard_colors=[ManimColor('#29ABCA')]
        )

        plot563.add(surface)
        self.play(Create(surface))

        self.next_slide()

        self.play(plot563.animate.scale(0.7).shift(OUT))


        equation534 = Tex(
            fr"$Price = {model.coef_[0]:,.1f} \cdot Size - {abs(model.coef_[1]):,.1f} \cdot Ghosts + {model.intercept_:,.1f}$")

        self.add_fixed_in_frame_mobjects(equation534)
        equation534.shift(DOWN * 3.25).scale(0.7)
        self.play(Write(equation534))
        
        self.next_slide()
        self.play(FadeOut(equation534), FadeOut(plot563))

        title5345 = Text("Beyond 3 Dimensions?").shift(UP * 3.35)
        self.play(ReplacementTransform(title6454, title5345))
        self.add_fixed_in_frame_mobjects(title5345)

        end = Text("It's Impossible to Visualize But We Can Do It Numerically.").scale(0.75)
        self.add_fixed_in_frame_mobjects(end)
        self.play(Write(end))

        


# class LinearRegressionMoreFeatures(ThreeDSlide):
#     def construct(self):
#         title = Text("How About Adding More Dimensions?").scale(
#             2).scale(0.5).shift(UP * 3.35)
#         self.add_fixed_in_frame_mobjects(title)
#         self.play(Write(title))
#         self.next_slide()
#         self.set_camera_orientation(phi=2*PI/5, theta=PI/5)
#         plot = VGroup()
#         ax = ThreeDAxes(
#             x_range=[0, 500, 100],
#             y_range=[0, 1, 5],
#             z_range=[0, 500000, 100000],
#             tips=True,
#             axis_config={"include_numbers": True},
#             x_length=7,
#             z_length=5).add_coordinates()
#         labels = ax.get_axis_labels(
#             Text("Size").scale(0.7), Text("# of Ghosts").scale(
#                 0.45), Text("Price").scale(0.45)
#         )

#         plot.add(ax, labels)

#         plot.shift([0, 0, -3])
#         plot.scale(0.8)

#         self.play(FadeIn(ax), FadeIn(labels))
#         self.next_slide()



#         dots = [Dot3D(point=ax.coords_to_point(*c), color=GREEN)
#                 for c in zip(size, ghosts, price)]

#         self.play(*[FadeIn(d) for d in dots])
#         self.next_slide()


# # class Main(Slide):
# #     def construct(self):
# #         intro(self)
# #         linear_regression(self)
